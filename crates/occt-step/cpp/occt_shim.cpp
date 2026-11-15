// Minimal C ABI over OpenCASCADE: read STEP / IGES / BREP into an XCAF
// document, tessellate it (in parallel) and expose meshes + instances.

#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IGESCAFControl_Reader.hxx>
#include <IMeshTools_Parameters.hxx>
#include <Message.hxx>
#include <Message_Messenger.hxx>
#include <Message_ProgressIndicator.hxx>
#include <Message_ProgressScope.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Poly_Polygon3D.hxx>
#include <Poly_Triangulation.hxx>
#include <RWMesh_FaceIterator.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <Standard_Failure.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <XCAFPrs_DocumentExplorer.hxx>
#include <XCAFPrs_Style.hxx>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <vector>

extern "C" {

struct OcctGroup {
    uint32_t first_index;
    uint32_t index_count;
    float rgba[4]; // linear RGB + alpha
    int32_t has_color;
};

struct OcctMeshData {
    const float* positions; // xyz per vertex
    const float* normals;   // xyz per vertex
    uint32_t vertex_count;
    const uint32_t* indices;
    uint32_t index_count;
    const OcctGroup* groups;
    uint32_t group_count;
    const float* edge_positions; // xyz pairs (line list)
    uint32_t edge_vertex_count;
};

struct OcctInstanceData {
    uint32_t mesh;
    double matrix[12]; // row-major 3x4
    const char* name;  // UTF-8, owned by the result
};

typedef int32_t (*OcctProgressFn)(void* user, double fraction);

} // extern "C"

namespace {

struct Group {
    std::vector<uint32_t> indices;
    float rgba[4] = {0, 0, 0, 1};
    bool has_color = false;
};

struct Mesh {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<uint32_t> indices;
    std::vector<OcctGroup> groups;
    std::vector<float> edges;
};

struct Instance {
    uint32_t mesh = 0;
    double matrix[12] = {};
    std::string name;
};

} // namespace

struct OcctResult {
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
};

namespace {

std::mutex g_occt_mutex;

void copy_error(char* dst, size_t len, const std::string& msg) {
    if (dst == nullptr || len == 0) return;
    size_t n = msg.size() < len - 1 ? msg.size() : len - 1;
    std::memcpy(dst, msg.data(), n);
    dst[n] = 0;
}

void silence_messages() {
    static std::once_flag once;
    std::call_once(once, [] {
        const Handle(Message_Messenger)& m = Message::DefaultMessenger();
        if (!m.IsNull()) m->ChangePrinters().Clear();
    });
}

/// Forwards OCCT progress to the Rust callback, mapped into [base, base+span].
class ShimProgress : public Message_ProgressIndicator {
public:
    ShimProgress(OcctProgressFn fn, void* user, std::atomic<bool>* cancelled, double base, double span)
        : fn_(fn), user_(user), cancelled_(cancelled), base_(base), span_(span) {}

    Standard_Boolean UserBreak() override { return cancelled_->load(); }

    void Show(const Message_ProgressScope&, const Standard_Boolean) override {
        if (fn_ != nullptr && fn_(user_, base_ + span_ * GetPosition()) != 0) cancelled_->store(true);
    }

private:
    OcctProgressFn fn_;
    void* user_;
    std::atomic<bool>* cancelled_;
    double base_, span_;
};

std::string label_name(const TDF_Label& label) {
    Handle(TDataStd_Name) name;
    if (!label.IsNull() && label.FindAttribute(TDataStd_Name::GetID(), name)) {
        TCollection_AsciiString ascii(name->Get());
        return std::string(ascii.ToCString());
    }
    return std::string();
}

std::string style_key(const XCAFPrs_Style& style) {
    if (!style.IsSetColorSurf()) return "-";
    const Quantity_ColorRGBA& c = style.GetColorSurfRGBA();
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%.4f,%.4f,%.4f,%.4f", c.GetRGB().Red(), c.GetRGB().Green(),
                  c.GetRGB().Blue(), c.Alpha());
    return buf;
}

void push_point(std::vector<float>& v, const gp_Pnt& p) {
    v.push_back((float)p.X());
    v.push_back((float)p.Y());
    v.push_back((float)p.Z());
}

/// Feature edges of a part: B-rep edges, without seams and degenerate edges.
void extract_edges(const TopoDS_Shape& shape, std::vector<float>& out) {
    TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
    TopExp::MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
    for (int i = 1; i <= edge_faces.Extent(); ++i) {
        const TopoDS_Edge& edge = TopoDS::Edge(edge_faces.FindKey(i));
        if (BRep_Tool::Degenerated(edge)) continue;
        const TopTools_ListOfShape& faces = edge_faces.FindFromIndex(i);
        bool drawn = false;
        bool seam = false;
        for (TopTools_ListOfShape::Iterator it(faces); it.More() && !drawn; it.Next()) {
            const TopoDS_Face& face = TopoDS::Face(it.Value());
            if (BRep_Tool::IsClosed(edge, face)) {
                seam = true;
                break;
            }
            TopLoc_Location loc;
            const Handle(Poly_Triangulation)& tri = BRep_Tool::Triangulation(face, loc);
            if (tri.IsNull()) continue;
            Handle(Poly_PolygonOnTriangulation) poly = BRep_Tool::PolygonOnTriangulation(edge, tri, loc);
            if (poly.IsNull() || poly->NbNodes() < 2) continue;
            const gp_Trsf trsf = loc.Transformation();
            for (int n = 1; n < poly->NbNodes(); ++n) {
                push_point(out, tri->Node(poly->Node(n)).Transformed(trsf));
                push_point(out, tri->Node(poly->Node(n + 1)).Transformed(trsf));
            }
            drawn = true;
        }
        if (drawn || seam) continue;
        // Free edge without a face triangulation: fall back to its 3D polygon.
        TopLoc_Location loc;
        Handle(Poly_Polygon3D) poly = BRep_Tool::Polygon3D(edge, loc);
        if (poly.IsNull() || poly->NbNodes() < 2) continue;
        const gp_Trsf trsf = loc.Transformation();
        const TColgp_Array1OfPnt& nodes = poly->Nodes();
        for (int n = nodes.Lower(); n < nodes.Upper(); ++n) {
            push_point(out, nodes(n).Transformed(trsf));
            push_point(out, nodes(n + 1).Transformed(trsf));
        }
    }
}

/// Triangulates one part definition (already meshed by BRepMesh) into a Mesh,
/// grouping triangles by their resolved face colour.
Mesh build_mesh(const TDF_Label& ref_label, const XCAFPrs_Style& style) {
    Mesh mesh;
    std::map<std::string, Group> groups;
    for (RWMesh_FaceIterator face(ref_label, TopLoc_Location(), true, style); face.More(); face.Next()) {
        if (face.IsEmptyMesh()) continue;
        const XCAFPrs_Style& fs = face.FaceStyle();
        if (!fs.IsVisible()) continue;
        const uint32_t base = (uint32_t)(mesh.positions.size() / 3);
        for (int n = face.NodeLower(); n <= face.NodeUpper(); ++n) {
            push_point(mesh.positions, face.NodeTransformed(n));
            const gp_Dir d = face.NormalTransformed(n);
            mesh.normals.push_back((float)d.X());
            mesh.normals.push_back((float)d.Y());
            mesh.normals.push_back((float)d.Z());
        }
        const std::string key = style_key(fs);
        Group& g = groups[key];
        if (fs.IsSetColorSurf()) {
            const Quantity_ColorRGBA& c = fs.GetColorSurfRGBA();
            // Quantity_Color stores linear RGB; the readers decode file colours as sRGB.
            g.rgba[0] = (float)c.GetRGB().Red();
            g.rgba[1] = (float)c.GetRGB().Green();
            g.rgba[2] = (float)c.GetRGB().Blue();
            g.rgba[3] = c.Alpha();
            g.has_color = true;
        }
        const int lower = face.NodeLower();
        for (int t = face.ElemLower(); t <= face.ElemUpper(); ++t) {
            int a, b, c;
            face.TriangleOriented(t).Get(a, b, c);
            g.indices.push_back(base + (uint32_t)(a - lower));
            g.indices.push_back(base + (uint32_t)(b - lower));
            g.indices.push_back(base + (uint32_t)(c - lower));
        }
    }
    for (auto& kv : groups) {
        OcctGroup og{};
        og.first_index = (uint32_t)mesh.indices.size();
        og.index_count = (uint32_t)kv.second.indices.size();
        std::memcpy(og.rgba, kv.second.rgba, sizeof(og.rgba));
        og.has_color = kv.second.has_color ? 1 : 0;
        mesh.indices.insert(mesh.indices.end(), kv.second.indices.begin(), kv.second.indices.end());
        mesh.groups.push_back(og);
    }
    extract_edges(XCAFDoc_ShapeTool::GetShape(ref_label), mesh.edges);
    return mesh;
}

void report(OcctProgressFn fn, void* user, std::atomic<bool>& cancelled, double fraction) {
    if (fn != nullptr && fn(user, fraction) != 0) cancelled.store(true);
}

/// Prints stage durations to stderr when VIEWER3D_TIMING is set (diagnostics).
struct StageTimer {
    bool enabled = std::getenv("VIEWER3D_TIMING") != nullptr;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    void stage(const char* name) {
        if (!enabled) return;
        auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[occt] %-10s %8.1f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
    }
};

OcctResult* load_impl(const char* path, int32_t format, double lin_rel, double angle, OcctProgressFn fn,
                      void* user, std::string& err) {
    std::atomic<bool> cancelled(false);
    StageTimer timer;
    silence_messages();

    Handle(XCAFApp_Application) app = XCAFApp_Application::GetApplication();
    Handle(TDocStd_Document) doc;
    app->NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc);
    struct DocCloser {
        Handle(XCAFApp_Application) app;
        Handle(TDocStd_Document) doc;
        ~DocCloser() {
            try {
                if (!doc.IsNull() && doc->IsOpened()) app->Close(doc);
            } catch (...) {
            }
        }
    } closer{app, doc};

    report(fn, user, cancelled, 0.0);
    if (format == 0 || format == 1) {
        Handle(ShimProgress) progress = new ShimProgress(fn, user, &cancelled, 0.35, 0.35);
        bool ok = false;
        if (format == 0) {
            STEPCAFControl_Reader reader;
            reader.SetColorMode(true);
            reader.SetNameMode(true);
            reader.SetLayerMode(false);
            reader.SetPropsMode(false);
            reader.SetGDTMode(false);
            reader.SetMatMode(false);
            reader.SetViewMode(false);
            if (reader.ReadFile(path) != IFSelect_RetDone) {
                err = "The STEP file could not be read (invalid or unsupported content).";
                return nullptr;
            }
            timer.stage("read");
            report(fn, user, cancelled, 0.35);
            if (cancelled.load()) { err = "cancelled"; return nullptr; }
            ok = reader.Transfer(doc, progress->Start());
            timer.stage("transfer");
        } else {
            IGESCAFControl_Reader reader;
            reader.SetColorMode(true);
            reader.SetNameMode(true);
            reader.SetLayerMode(false);
            if (reader.ReadFile(path) != IFSelect_RetDone) {
                err = "The IGES file could not be read (invalid or unsupported content).";
                return nullptr;
            }
            report(fn, user, cancelled, 0.35);
            if (cancelled.load()) { err = "cancelled"; return nullptr; }
            ok = reader.Transfer(doc, progress->Start());
        }
        if (cancelled.load()) { err = "cancelled"; return nullptr; }
        if (!ok) {
            err = "The file contains no geometry that could be converted.";
            return nullptr;
        }
    } else {
        TopoDS_Shape shape;
        BRep_Builder builder;
        if (!BRepTools::Read(shape, path, builder) || shape.IsNull()) {
            err = "The BREP file could not be read.";
            return nullptr;
        }
        XCAFDoc_DocumentTool::ShapeTool(doc->Main())->AddShape(shape);
    }
    report(fn, user, cancelled, 0.7);

    Handle(XCAFDoc_ShapeTool) shape_tool = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    TDF_LabelSequence roots;
    shape_tool->GetFreeShapes(roots);
    TopoDS_Compound all;
    BRep_Builder builder;
    builder.MakeCompound(all);
    for (TDF_LabelSequence::Iterator it(roots); it.More(); it.Next()) {
        TopoDS_Shape s = XCAFDoc_ShapeTool::GetShape(it.Value());
        if (!s.IsNull()) builder.Add(all, s);
    }
    Bnd_Box box;
    BRepBndLib::Add(all, box);
    if (box.IsVoid()) {
        err = "The file contains no surfaces to display.";
        return nullptr;
    }
    const double diag = std::sqrt(box.SquareExtent());

    IMeshTools_Parameters params;
    params.Deflection = diag * lin_rel;
    params.Angle = angle;
    params.Relative = Standard_False;
    params.InParallel = Standard_True;
    params.MinSize = params.Deflection * 0.1;
    {
        Handle(ShimProgress) progress = new ShimProgress(fn, user, &cancelled, 0.7, 0.25);
        BRepMesh_IncrementalMesh mesher(all, params, progress->Start());
    }
    timer.stage("mesh");
    if (cancelled.load()) { err = "cancelled"; return nullptr; }
    report(fn, user, cancelled, 0.95);

    OcctResult* result = new OcctResult();
    std::map<std::string, uint32_t> mesh_cache;
    for (XCAFPrs_DocumentExplorer expl(doc, XCAFPrs_DocumentExplorerFlags_OnlyLeafNodes); expl.More(); expl.Next()) {
        const XCAFPrs_DocumentNode& node = expl.Current();
        if (!node.Style.IsVisible()) continue;
        TCollection_AsciiString entry;
        TDF_Tool::Entry(node.RefLabel, entry);
        const std::string key = std::string(entry.ToCString()) + "|" + style_key(node.Style);
        uint32_t mesh_index;
        auto found = mesh_cache.find(key);
        if (found != mesh_cache.end()) {
            mesh_index = found->second;
        } else {
            Mesh mesh = build_mesh(node.RefLabel, node.Style);
            if (mesh.indices.empty() && mesh.edges.empty()) {
                mesh_cache[key] = UINT32_MAX;
                continue;
            }
            mesh_index = (uint32_t)result->meshes.size();
            result->meshes.push_back(std::move(mesh));
            mesh_cache[key] = mesh_index;
        }
        if (mesh_index == UINT32_MAX) continue;

        Instance inst;
        inst.mesh = mesh_index;
        const gp_Trsf trsf = node.Location.Transformation();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) inst.matrix[r * 4 + c] = trsf.Value(r + 1, c + 1);
        inst.name = label_name(node.Label);
        if (inst.name.empty()) inst.name = label_name(node.RefLabel);
        result->instances.push_back(std::move(inst));
        if (cancelled.load()) {
            delete result;
            err = "cancelled";
            return nullptr;
        }
    }
    if (result->instances.empty()) {
        delete result;
        err = "The file contains no surfaces to display.";
        return nullptr;
    }
    timer.stage("extract");
    report(fn, user, cancelled, 1.0);
    return result;
}

} // namespace

extern "C" {

/// format: 0 = STEP, 1 = IGES, 2 = BREP. `lin_rel` is the chordal deflection
/// relative to the model's bounding-box diagonal, `angle` in radians.
OcctResult* occt_shim_load(const char* path, int32_t format, double lin_rel, double angle, OcctProgressFn fn,
                           void* user, char* err, size_t err_len) {
    std::lock_guard<std::mutex> lock(g_occt_mutex);
    std::string message;
    try {
        OcctResult* r = load_impl(path, format, lin_rel, angle, fn, user, message);
        if (r == nullptr) copy_error(err, err_len, message.empty() ? std::string("unknown error") : message);
        return r;
    } catch (const Standard_Failure& f) {
        copy_error(err, err_len, std::string("OpenCASCADE error: ") + f.GetMessageString());
    } catch (const std::exception& e) {
        copy_error(err, err_len, std::string("error: ") + e.what());
    } catch (...) {
        copy_error(err, err_len, "unknown OpenCASCADE failure");
    }
    return nullptr;
}

uint32_t occt_shim_mesh_count(const OcctResult* r) { return (uint32_t)r->meshes.size(); }

OcctMeshData occt_shim_mesh(const OcctResult* r, uint32_t i) {
    const Mesh& m = r->meshes[i];
    OcctMeshData d{};
    d.positions = m.positions.data();
    d.normals = m.normals.data();
    d.vertex_count = (uint32_t)(m.positions.size() / 3);
    d.indices = m.indices.data();
    d.index_count = (uint32_t)m.indices.size();
    d.groups = m.groups.data();
    d.group_count = (uint32_t)m.groups.size();
    d.edge_positions = m.edges.data();
    d.edge_vertex_count = (uint32_t)(m.edges.size() / 3);
    return d;
}

uint32_t occt_shim_instance_count(const OcctResult* r) { return (uint32_t)r->instances.size(); }

OcctInstanceData occt_shim_instance(const OcctResult* r, uint32_t i) {
    const Instance& inst = r->instances[i];
    OcctInstanceData d{};
    d.mesh = inst.mesh;
    std::memcpy(d.matrix, inst.matrix, sizeof(d.matrix));
    d.name = inst.name.c_str();
    return d;
}

void occt_shim_free(OcctResult* r) { delete r; }

} // extern "C"
