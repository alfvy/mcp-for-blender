"""Regression coverage for the Blendkit integration.

The traps this file guards against were all found against the live API, not read
from the Blendkit addon's source - which is worth saying, because the addon does
not talk to blendkit.com at all. Every request it makes for search or download
goes to a local Go daemon (`bk_client`), which is an uninitialised git submodule
in a checkout of it, so there is nothing to call and no response shape to copy.

The behaviours pinned here:

* `is_free:true` is not a filter. With it, with `is_free:false` and with no
  filter at all, the API returns byte-identical results and the same count, so
  the free plan has to be applied to the response's own `access` field.
* `downloadUrl` is not a file. It is a resolution endpoint that answers
  `{"fileType", "uuid", "filePath"}`, and it 403s with "Parameter scene_uuid
  not set" when that parameter is absent - a body that reads exactly like an
  authentication failure and is not one.
* The `filePath` it returns is signed with a short-lived `?verify=` token, so it
  cannot be cached or reused across calls.
* A Full Plan asset answers the same resolve call with 401 "Please log in",
  even though search handed back a perfectly usable downloadUrl for it.
* `is_free:false` and a paid asset are indistinguishable in a search result
  without reading `access`, and paid assets are a large share of the library.

Every request here is mocked; the suite never touches the network.
"""
import importlib.util
import re
import sys
import types

import pytest

from conftest import ROOT_ADDON as ADDON


# --- a fake Blender, scoped to what the Blendkit paths touch -------------------

class FakeObject:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.type = "MESH"
        self.material_slots = []
        self.custom_properties = {}

    def __setitem__(self, key, value):
        self.custom_properties[key] = value


class FakeObjects(list):
    def get(self, name):
        return next((o for o in self if o.name == name), None)

    def link(self, obj):
        self.append(obj)


class FakeCollections(list):
    def new(self, name):
        collection = FakeCollection(name)
        self.append(collection)
        return collection

    def link(self, collection):
        self.append(collection)


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.objects = FakeObjects()
        self.children = FakeCollections()


class FakeMaterial:
    def __init__(self, name):
        self.name = name
        self.use_nodes = False
        self.custom_properties = {}

    def __setitem__(self, key, value):
        self.custom_properties[key] = value

    def __getitem__(self, key):
        return self.custom_properties[key]

    def get(self, key, default=None):
        return self.custom_properties.get(key, default)


class FakeMaterials(list):
    def new(self, name):
        material = FakeMaterial(name)
        self.append(material)
        return material

    def get(self, name):
        return next((m for m in self if m.name == name), None)


class FakeImage:
    def __init__(self, name):
        self.name = name
        self.packed_file = None
        self.pack_calls = 0

    def pack(self):
        self.packed_file = object()
        self.pack_calls += 1


class FakeImages(list):
    def get(self, name):
        return next((i for i in self if i.name == name), None)

    def load(self, filepath, check_existing=False):
        if check_existing:
            for existing in self:
                if existing.filepath == filepath:
                    return existing
        image = FakeImage(filepath)
        image.filepath = filepath
        self.append(image)
        return image


class FakeWorld:
    def __init__(self, name):
        self.name = name
        self.use_nodes = False
        self.custom_properties = {}
        self.node_tree = FakeNodeTree()

    def __setitem__(self, key, value):
        self.custom_properties[key] = value

    def get(self, key, default=None):
        return self.custom_properties.get(key, default)


class FakeWorlds(list):
    def new(self, name):
        world = FakeWorld(name)
        self.append(world)
        return world

    def get(self, name):
        return next((w for w in self if w.name == name), None)


class FakeNodeTree:
    def __init__(self):
        self.nodes = FakeNodes()
        self.links = FakeLinks()


class FakeNode:
    def __init__(self, bl_idname):
        self.bl_idname = bl_idname
        self.type = {"ShaderNodeBackground": "BACKGROUND",
                     "ShaderNodeTexEnvironment": "TEX_ENVIRONMENT",
                     "ShaderNodeTexCoord": "TEX_COORD",
                     "ShaderNodeMapping": "MAPPING",
                     "ShaderNodeOutputWorld": "OUTPUT_WORLD"}.get(bl_idname, "UNKNOWN")
        self.name = bl_idname
        self.location = (0, 0)
        self.image = None
        self.inputs = FakeSocketCollection(self, [])
        self.outputs = FakeSocketCollection(self, [])


class FakeNodes(list):
    def new(self, type=None):
        node = FakeNode(type)
        self.append(node)
        return node

    def clear(self):
        del self[:]


class FakeSocket:
    def __init__(self, node, name):
        self.node = node
        self.name = name
        self.default_value = None


class FakeSocketCollection:
    def __init__(self, node, names):
        self.node = node
        self._names = list(names)

    def __getitem__(self, key):
        return FakeSocket(self.node, key)

    def __contains__(self, key):
        return True

    def __iter__(self):
        return iter(FakeSocket(self.node, n) for n in self._names)


class FakeLink:
    def __init__(self, from_socket, to_socket):
        self.from_socket = from_socket
        self.to_socket = to_socket


class FakeLinks(list):
    def new(self, from_socket, to_socket):
        link = FakeLink(from_socket, to_socket)
        self.append(link)
        return link


class FakeLibraryLoad:
    """bpy.data.libraries.load: names in, appended datablocks out."""

    def __init__(self, data, contents, decorate=None):
        self._data = data
        self._contents = contents
        self._decorate = decorate or (lambda _name, obj: obj)

    def __enter__(self):
        self.data_from = types.SimpleNamespace(
            collections=list(self._contents.get("collections", {})),
            objects=list(self._contents.get("objects", [])),
            materials=list(self._contents.get("materials", [])),
            node_groups=list(self._contents.get("node_groups", [])),
        )
        self.data_to = types.SimpleNamespace(
            collections=[], objects=[], materials=[], node_groups=[])
        return self.data_from, self.data_to

    def __exit__(self, *_exc):
        appended = []
        for name in self.data_to.collections:
            collection = self._data.collections.new(name)
            for obj_name in self._contents["collections"][name]:
                obj = self._decorate(obj_name, FakeObject(obj_name))
                self._data.objects.append(obj)
                collection.objects.append(obj)
            appended.append(collection)
        self.data_to.collections = appended

        objects = []
        for name in self.data_to.objects:
            obj = self._decorate(name, FakeObject(name))
            self._data.objects.append(obj)
            objects.append(obj)
        self.data_to.objects = objects

        self.data_to.materials = [
            self._data.materials.new(name) for name in self.data_to.materials]
        self.data_to.node_groups = [
            FakeNodeGroup(name) for name in self.data_to.node_groups]
        return False


class FakeNodeGroup:
    def __init__(self, name):
        self.name = name
        self.nodes = []


class FakeLibraries:
    def __init__(self, data):
        self._data = data
        self.contents = {}
        # Lets a test hand each appended object material slots, the way a real
        # .blend does - which is what the image-packing path walks.
        self.decorate = None

    def load(self, filepath, link=False):
        assert link is False, "Blendkit assets are appended, not linked"
        return FakeLibraryLoad(self._data, self.contents, self.decorate)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", streamed=False, headers=None):
        self.status_code = status_code
        self._payload = payload
        self._content = content
        self._streamed = streamed
        self.headers = dict(headers or {})

    @property
    def content(self):
        # A streamed body must be consumed with iter_content. Serving .content
        # too would let the whole-file-into-memory path - a 948MB HDR - pass.
        if self._streamed:
            raise AssertionError(
                "a streamed download must use iter_content(), not response.content")
        return self._content

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _load_addon(monkeypatch, blender_version=(4, 2, 0)):
    bpy = types.ModuleType("bpy")

    bpy.data = types.SimpleNamespace(
        objects=FakeObjects(),
        collections=FakeCollections(),
        materials=FakeMaterials(),
        images=FakeImages(),
        worlds=FakeWorlds(),
    )
    bpy.data.libraries = FakeLibraries(bpy.data)

    scene = types.SimpleNamespace(
        blendermcp_use_blendkit=True,
        blendermcp_use_polyhaven=False,
        blendermcp_use_hyper3d=False,
        blendermcp_use_hunyuan3d=False,
        blendermcp_use_sketchfab=False,
        blendermcp_use_polypizza=False,
    )
    bpy.context = types.SimpleNamespace(
        scene=scene,
        selected_objects=[],
        collection=types.SimpleNamespace(objects=types.SimpleNamespace(link=lambda _o: None)),
        view_layer=types.SimpleNamespace(
            update=lambda: None,
            objects=types.SimpleNamespace(active=None),
        ),
    )
    # The scene's own collection, which imported collections are linked into.
    scene.collection = types.SimpleNamespace(
        children=FakeCollections(), objects=FakeObjects())
    bpy.context.collection = scene.collection
    bpy.ops = types.SimpleNamespace(
        import_scene=types.SimpleNamespace(gltf=lambda **_kwargs: None)
    )
    bpy.types = types.SimpleNamespace(
        AddonPreferences=object,
        Operator=object,
        Panel=object,
        Scene=type("Scene", (), {}),
    )

    props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty", "StringProperty"):
        setattr(props, name, lambda **_kwargs: None)
    bpy.props = props

    handlers = types.ModuleType("bpy.app.handlers")
    handlers.persistent = lambda fn: fn
    handlers.undo_post = []
    handlers.redo_post = []
    handlers.depsgraph_update_post = []

    app = types.ModuleType("bpy.app")
    app.version = blender_version
    app.version_string = f"{blender_version[0]}.{blender_version[1]}.0"
    app.background = False
    app.handlers = handlers
    app.timers = types.SimpleNamespace(
        is_registered=lambda *_a, **_k: False,
        register=lambda *_a, **_k: None,
        unregister=lambda *_a, **_k: None,
    )
    bpy.app = app

    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    monkeypatch.setitem(sys.modules, "bpy.app", app)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers)
    monkeypatch.setitem(sys.modules, "mathutils", types.ModuleType("mathutils"))

    requests = types.ModuleType("requests")
    requests.utils = types.SimpleNamespace(default_headers=dict)
    requests.exceptions = types.SimpleNamespace(Timeout=TimeoutError)
    monkeypatch.setitem(sys.modules, "requests", requests)

    spec = importlib.util.spec_from_file_location("blender_mcp_blendkit_test", ADDON)
    addon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(addon)
    return addon


@pytest.fixture
def server(monkeypatch, tmp_path):
    addon = _load_addon(monkeypatch)
    # Downloads go to a fresh mkdtemp the addon removes on the way out; rooting
    # it under tmp_path makes a leaked directory visible to the tests.
    monkeypatch.setattr(addon.tempfile, "gettempdir", lambda: str(tmp_path))
    # One uuid for the session, so the scene_uuid assertions below are stable.
    addon._blendkit_scene_uuid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    srv = addon.BlenderMCPServer()
    # A model .blend by default, so a bare download has something to append.
    # An empty one is now an error, which several tests below rely on.
    addon.bpy.data.libraries.contents = dict(BLEND_CONTENTS)
    return addon, srv


# --- fixtures in the shape the live API returns -------------------------------

SEARCH_URL = "https://www.blendkit.com/api/v1/search/"


def _file(file_type, uuid, size=1024, resolution=None):
    return {
        "fileType": file_type,
        "id": 1,
        "downloadUrl": f"https://www.blendkit.com/api/v1/downloads/{uuid}/",
        "uuid": uuid,
        "created": "2025-08-04T13:25:00.434054+02:00",
        "isThumbnail": False,
        "fileUploadSize": size,
        "filename": f"assets/x/files/{file_type}_{uuid}.blend",
        "resolution": resolution,
    }


BLEND_UUID = "ad331a6d-22a9-4987-ade1-afc7755a5561"
GLTF_UUID = "bac3d041-13c0-49b1-b5c7-786f6a6f6e8d"

THUMB = ("https://public.blenderkit.com/thumbnails/assets/"
         "603d2a33426b4690898213520b58d162/files/"
         "thumbnail_63e34dfa-fba6-49bb-876b-72fb3a1b7731.jpg"
         ".256x256_q85_crop-,.jpg")

TREE = {
    "id": "603d2a33-426b-4690-8982-13520b58d162",
    "assetBaseId": "ea8df77e-35d8-4624-8825-b3f3f5b3d3b2",
    "name": "A tree",
    "displayName": "Tree",
    "description": "A basic tree that can be used in any scene",
    "assetType": "model",
    "category": "tree",
    "tags": ["tree", "nature", "green"],
    "license": "royalty_free",
    "author": {"id": 588525, "fullName": "Toby Noby"},
    "access": "free",
    "accessLabel": "Free",
    "canDownload": True,
    "isFree": True,
    "verificationStatus": "validated",
    # Older than the 4.2 the fake Blender reports, so it is importable. The
    # newer-blend cases override this.
    "sourceAppVersion": "4.1.0",
    "filesSize": 436770333,
    "files": [
        _file("blend", BLEND_UUID, size=83469545),
        _file("gltf", GLTF_UUID, size=15853120),
        _file("resolution_0_5K", "11111111-1111-1111-1111-111111111111", size=740312),
        _file("resolution_1K", "22222222-2222-2222-2222-222222222222", size=2954591),
        _file("resolution_2K", "33333333-3333-3333-3333-333333333333", size=11804915),
        _file("thumbnail", "44444444-4444-4444-4444-444444444444", size=671289),
    ],
    "thumbnailSmallUrl": THUMB,
    "thumbnailLargeUrl": THUMB.replace("256x256", "1024x1024"),
    "dictParameters": {
        "faceCount": 1439949,
        "dimensionX": 9.4789,
        "dimensionY": 9.7286,
        "dimensionZ": 14.1296,
        "textureResolutionMax": 2048,
        "objectCount": 2,
    },
}

# The other half of a typical page: a Full Plan asset with a downloadUrl that
# would fail. Offering it would cost the caller a round trip and a confusing 401.
CAR = {
    "id": "603d2a33-426b-4690-8982-13520b58d999",
    "assetBaseId": "ea8df77e-35d8-4624-8825-b3f3f5b3d999",
    "name": "Classic car in parking lot",
    "assetType": "model",
    "license": "royalty_free",
    "author": {"id": 1, "fullName": "Someone"},
    "access": "full",
    "accessLabel": "Full Plan",
    "canDownload": False,
    "isFree": False,
    "sourceAppVersion": "4.2.0",
    "files": [_file("blend", "55555555-5555-5555-5555-555555555555")],
    "thumbnailSmallUrl": THUMB,
    "dictParameters": {"faceCount": 900000},
}

PAGE = {"count": 40, "results": [TREE, CAR]}

# What the downloaded .blend holds: the asset's collection, a LOD chain, and a
# second unrelated model stowaged in the same file.
BLEND_CONTENTS = {
    "collections": {
        "A tree": ["A tree_trunk"],
        "A tree_LOD0": ["A tree_trunk_LOD0", "A tree_leaves_LOD0"],
        "someone_elses_model": ["someone_elses_model"],
    },
    "objects": ["A tree_trunk", "A tree_trunk_LOD0", "A tree_leaves_LOD0",
                "someone_elses_model"],
}

RESOLVED = {
    "fileType": "blend",
    "uuid": BLEND_UUID,
    "filePath": "https://assets.blenderkit.com/assets/603d2a/files/blend_x.blend?verify=1791136652-hmac",
}


# The most recent request log, for the tests that need to assert on call
# ordering across a helper boundary. Set by _install_requests.
_LAST_CALLS = [None]


def _install_requests(monkeypatch, addon, page=None, resolve=None,
                      thumb_body=b"\xff\xd8\xff\xe0jpeg", file_bytes=b"blend-bytes",
                      detail=None):
    """Route requests by URL, and record every one of them."""
    calls = []
    page = PAGE if page is None else page
    resolve = RESOLVED if resolve is None else resolve
    # What /assets/<version id>/ answers. Defaults to the same two assets search
    # returns, so a test that only cares about one can pass detail=[...].
    detail = [TREE, CAR] if detail is None else detail

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        calls.append({
            "url": url,
            "params": dict(params or {}),
            "stream": stream,
            "timeout": timeout,
            "headers": dict(headers or {}),
        })
        if stream:
            # The bytes have to look like a .blend this Blender can open, or the
            # import falls back to glTF and the append never happens. 4.2, to
            # match the fake Blender, and not compressed, so the header parser
            # reads it without needing zstandard.
            return FakeResponse(
                content=file_bytes if file_bytes != b"blend-bytes"
                else b"BLENDER-v402" + b"\x00" * 32,
                streamed=True,
            )
        if "/downloads/" in url:
            # The resolve endpoint answers per requested file: the fileType it
            # reports is the one whose uuid was asked for, which is what makes
            # the glTF fallback and the resolution ladder testable at all.
            uuid = url.split("/downloads/")[-1].split("/")[0]
            if isinstance(resolve, dict) and resolve.get("uuid") == uuid:
                return FakeResponse(payload=resolve)
            for entry in (TREE["files"] + CAR["files"]
                          + [f for a in detail for f in a.get("files", [])]):
                if entry["uuid"] == uuid:
                    return FakeResponse(payload={
                        "fileType": entry["fileType"], "uuid": uuid,
                        "filePath": f"https://assets.blenderkit.com/{uuid}.blend?verify=1-hmac",
                    })
            return FakeResponse(payload=resolve)
        if "/api/v1/assets/" in url:
            # The detail endpoint, keyed on the *version* id. It serves the same
            # files as search but omits canDownload, which is what makes the
            # entitlement check in download meaningful here. Matched on the API
            # path, not on "/assets/", because a thumbnail URL contains that too.
            ident = url.split("/api/v1/assets/")[-1].split("/")[0]
            for candidate in detail:
                if candidate["id"] == ident:
                    return FakeResponse(payload={
                        k: v for k, v in candidate.items() if k != "canDownload"})
            return FakeResponse(status_code=404, payload={"detail": "Not found."})
        if "/search/" in url:
            return FakeResponse(payload=page)
        return FakeResponse(content=thumb_body, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)
    _LAST_CALLS[0] = calls
    return calls


def _downloaded(calls):
    return [call["url"] for call in calls if call["stream"]]


def _search_calls(calls):
    return [call for call in calls if "/search/" in call["url"]]


def _resolve_calls(calls):
    return [call for call in calls if "/downloads/" in call["url"]]


# --- the free plan is a response field, not a query filter --------------------

def test_paid_assets_are_filtered_out_of_results(server, monkeypatch):
    """`is_free:true` looks like the filter for this and is not - the API returns
    byte-identical results with it, with `is_free:false` and with nothing at all.
    Access is a per-asset field, so a Full Plan asset has to be dropped here or
    it gets offered and then fails with a 401 on download."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    result = srv.search_blendkit_assets("tree")

    assert "error" not in result, result
    ids = [a["id"] for a in result["assets"]]
    assert ids == [TREE["id"]]
    assert CAR["id"] not in ids
    assert result["free_count"] == 1
    # And the count the user is shown distinguishes free from total, rather than
    # presenting a full-plan result as if it were available.
    assert result["total_count"] == 40


def test_the_free_count_is_not_presented_as_library_coverage(server, monkeypatch):
    """`free_count` is how many free results came back in the page fetched, not
    how many exist in the library. Reporting "13 of 4608" reads as a coverage
    figure and is badly wrong - the true free total needs the whole library
    paged, and for scarce types the free few sit deep in the ranking."""
    addon, srv = server
    page = {"count": 4608, "results": [TREE, CAR] * 25}
    _install_requests(monkeypatch, addon, page=page)

    result = srv.search_blendkit_assets("tree")

    assert result["free_count"] == 25
    assert result["checked_count"] == 50
    assert result["total_count"] == 4608


def test_no_is_free_parameter_is_sent(server, monkeypatch):
    """Belt and braces on the above: if a future change does add `is_free:true`,
    this fails rather than shipping a filter that silently does nothing."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("tree")

    query = _search_calls(calls)[0]["params"]["query"]
    assert "is_free" not in query


def test_a_paid_asset_cannot_be_downloaded_even_when_named_directly(server, monkeypatch):
    """Download looks the asset up by id rather than trusting the caller to pass
    something a search returned, so the access check has to be here too - and
    the resolve is never attempted."""
    addon, srv = server
    _install_requests(monkeypatch, addon, page={"count": 1, "results": [CAR]})
    calls = _LAST_CALLS[0]

    result = srv.download_blendkit_asset(CAR["id"])

    assert "error" in result
    assert "Full Plan" in result["error"]
    assert _resolve_calls(calls) == []
    assert _downloaded(calls) == []


def test_a_page_with_no_free_assets_says_so_instead_of_looking_empty(server, monkeypatch):
    """"No results" reads as "Blendkit has none of this", which is a much worse
    and quite wrong statement when it has forty that need a paid plan."""
    addon, srv = server
    _install_requests(monkeypatch, addon, page={"count": 12, "results": [CAR]})

    result = srv.search_blendkit_assets("sports car")

    assert result["assets"] == []
    assert "None of the 12 matching" in result["note"]
    assert "free plan" in result["note"]


def test_an_empty_page_with_a_count_still_explains_itself(server, monkeypatch):
    """A page can come back empty while the count is non-zero - the server cut
    the page, and everything on it was filtered out here. Staying silent would
    read as "Blendkit has nothing"."""
    addon, srv = server
    _install_requests(monkeypatch, addon, page={"count": 1, "results": []})

    result = srv.search_blendkit_assets("skin", asset_type="nodegroup")

    assert result["returned_count"] == 0
    assert result["asset_type"] == "nodegroup"
    assert "None of the 1 matching" in result["note"]
    assert "node groups" in result["note"]


# --- the query string --------------------------------------------------------

def test_filters_are_terms_inside_one_query_string(server, monkeypatch):
    """Not query parameters: Blendkit takes a single Lucene-ish `query`, and
    every filter is a +field:value term inside it. Sending them as separate
    params would return the unfiltered page."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("stone wall", asset_type="material", max_face_count=5000)

    params = _search_calls(calls)[0]["params"]
    query = params["query"]
    # Space-encoded, as Blendkit's own query builder does it.
    assert "stone+wall" in query
    assert "+asset_type:material" in query
    assert "+faceCount_lte:5000" in query
    # The real params, alongside the query.
    assert params["dict_parameters"] == 1
    assert params["scene_uuid"] == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def test_keywords_are_url_quoted(server, monkeypatch):
    """A space or an ampersand in the keywords would otherwise truncate the query
    at the first space or split it into a second, meaningless term."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("rock & stone wall")

    query = _search_calls(calls)[0]["params"]["query"]
    assert "rock+%26+stone+wall" in query


def test_blender_version_is_padded_so_it_sorts_numerically(server, monkeypatch):
    """Blendkit sorts these as numbers, so an unpadded 4.10 sorts below 4.9 and
    the filter excludes the very version it names."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("tree", blender_version=(4, 10))

    query = _search_calls(calls)[0]["params"]["query"]
    assert "+sourceAppVersion_gte:4.010" in query
    assert "+sourceAppVersion_lt:5.010" in query


def test_an_omitted_query_still_searches(server, monkeypatch):
    """Browsing is a real use - Blendkit's own addon opens with an empty query.
    The asset_type term alone is a valid query, so it is what gets sent; the
    `*` fallback covers the case where every term was dropped."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.search_blendkit_assets()

    query = _search_calls(calls)[0]["params"]["query"]
    assert query == "+asset_type:model"
    assert result["query"] is None


def test_the_query_falls_back_to_match_all_when_it_would_be_empty(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("   ")

    assert _search_calls(calls)[0]["params"]["query"] == "+asset_type:model"


def test_an_unknown_asset_type_is_rejected_without_a_request(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.search_blendkit_assets("tree", asset_type="furniture")

    assert "error" in result
    assert "Invalid asset type" in result["error"]
    assert calls == []


def test_a_non_numeric_face_filter_is_rejected(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.search_blendkit_assets("tree", max_face_count="lots")

    assert "error" in result
    assert calls == []


def test_limit_is_clamped(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon)

    assert srv.search_blendkit_assets(limit=5)["returned_count"] == 1
    assert srv.search_blendkit_assets(limit=5000)["returned_count"] == 1
    assert srv.search_blendkit_assets(limit=0)["returned_count"] == 1


def test_over_fetches_so_filtering_cannot_empty_the_page(server, monkeypatch):
    """Filtering only ever shrinks a page the server already cut, so the matches
    that survive would be the ones further down it. 40 results are asked for to
    return 20."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets(limit=20)

    assert _search_calls(calls)[0]["params"]["page_size"] == 50


# --- the download call chain -------------------------------------------------

def test_the_download_url_is_resolved_with_a_scene_uuid(server, monkeypatch):
    """Without scene_uuid the resolve endpoint answers 403 "Parameter scene_uuid
    not set" - a body that reads exactly like an auth failure and is not one."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    resolved = _resolve_calls(calls)[0]["url"]
    assert "scene_uuid=3f2504e0-4f89-11d3-9a0c-0305e82c3301" in resolved


def test_scene_uuid_is_appended_not_interleaved(server, monkeypatch):
    """A resolve URL rebuilt with urlparse would drop the endpoint's own trailing
    slash or reorder its parameters; appending cannot."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.download_blendkit_asset(TREE["id"])

    url = _resolve_calls(calls)[0]["url"]
    assert url.startswith("https://www.blendkit.com/api/v1/downloads/")
    assert "/?" in url
    assert url.endswith("?scene_uuid=3f2504e0-4f89-11d3-9a0c-0305e82c3301")


def test_a_resolved_url_is_never_cached(server, monkeypatch):
    """The filePath Blendkit returns is signed with a `?verify=<epoch>-<hmac>`
    token whose epoch is the moment of the request, so it is not reusable."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.download_blendkit_asset(TREE["id"])
    srv.download_blendkit_asset(TREE["id"])

    # Two downloads, two resolves - not one resolve reused.
    assert len(_resolve_calls(calls)) == 2


def test_the_scene_uuid_is_stable_within_a_session(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.download_blendkit_asset(TREE["id"])
    srv.download_blendkit_asset(TREE["id"])
    srv.search_blendkit_assets("tree")

    uuids = {_blendkit_scene_id(call["url"]) for call in calls if "scene_uuid" in call["url"]}
    assert len(uuids) == 1


def _blendkit_scene_id(url):
    return re.search(r"scene_uuid=([^&]+)", url).group(1)


def test_downloads_are_streamed_not_buffered(server, monkeypatch):
    """A free HDR at 8K is 39MB and the largest .blend I measured 948MB, all on
    Blender's main thread. .content would hold every byte in memory."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.download_blendkit_asset(TREE["id"])

    assert len(_downloaded(calls)) == 1


def test_every_request_carries_a_timeout(server, monkeypatch):
    """Without one, a stalled connection hangs Blender's main thread forever -
    there is no progress bar and no way to cancel."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.download_blendkit_asset(TREE["id"])

    assert calls
    assert all(call["timeout"] is not None for call in calls)


def test_the_downloaded_file_is_cleaned_up(server, monkeypatch, tmp_path):
    """Appended images reference their files by path. Those paths are in a
    temporary directory, so it is removed only after they are packed - see the
    packing test - and must not be left behind."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    assert srv.download_blendkit_asset(TREE["id"]).get("success")
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == []


def test_the_lod0_collection_is_appended_not_every_lod(server, monkeypatch):
    """A model with levels of detail carries all of them in one file. Appending
    them wholesale put three copies of the model on top of each other."""
    addon, srv = server
    _install_requests(monkeypatch, addon)
    addon.bpy.data.libraries.contents = BLEND_CONTENTS

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert set(result["imported_objects"]) == {"A tree_trunk_LOD0", "A tree_leaves_LOD0"}
    assert [c.name for c in addon.bpy.context.scene.collection.children] == ["A tree_LOD0"]


def test_a_second_model_in_the_same_file_is_left_out(server, monkeypatch):
    """Blendkit .blends sometimes hold an unrelated model beside the asset."""
    addon, srv = server
    _install_requests(monkeypatch, addon)
    addon.bpy.data.libraries.contents = BLEND_CONTENTS

    result = srv.download_blendkit_asset(TREE["id"])

    assert "someone_elses_model" not in result["imported_objects"]


def test_a_blend_with_no_matching_collection_still_imports(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon)
    addon.bpy.data.libraries.contents = {
        "collections": {"Collection": ["loose_mesh"]},
        "objects": ["loose_mesh"],
    }

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert result["imported_objects"] == ["loose_mesh"]


def test_imported_images_are_packed(server, monkeypatch):
    """Appended images point at the temporary directory, which is removed on the
    way out - so without packing, reopening the .blend loses its textures."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    # An appended Blendkit .blend arrives with materials already wired to images
    # on disk - paths inside the temporary directory this import is about to
    # delete. That is what has to be packed.
    image = FakeImage("tree_diffuse.png")
    material = FakeMaterial("Bark")
    material.use_nodes = True
    material.node_tree = types.SimpleNamespace(
        nodes=[types.SimpleNamespace(type="TEX_IMAGE", image=image)], links=[])

    def decorate(_name, obj):
        obj.material_slots = [types.SimpleNamespace(material=material)]
        return obj

    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": ["A tree_trunk"]}
    addon.bpy.data.libraries.decorate = decorate

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert image.packed_file is not None
    assert result["materials"] == ["Bark"]


def test_a_non_image_texture_node_is_left_alone(server, monkeypatch):
    """A procedural material has nodes but nothing to pack, and walking it must
    not fail or invent an image."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    material = FakeMaterial("Procedural")
    material.use_nodes = True
    material.node_tree = types.SimpleNamespace(
        nodes=[types.SimpleNamespace(type="TEX_IMAGE", image=None)], links=[])

    def decorate(_name, obj):
        obj.material_slots = [types.SimpleNamespace(material=material)]
        return obj

    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": ["A tree_trunk"]}
    addon.bpy.data.libraries.decorate = decorate

    assert srv.download_blendkit_asset(TREE["id"]).get("success")


def test_provenance_is_stamped_on_imported_datablocks(server, monkeypatch):
    """Blendkit's free tier is royalty-free rather than CC0, so there is nothing
    to attribute - but the origin still has to outlive the session, and a custom
    property is what is saved into the .blend."""
    addon, srv = server
    _install_requests(monkeypatch, addon)
    addon.bpy.data.libraries.contents = BLEND_CONTENTS

    result = srv.download_blendkit_asset(TREE["id"])

    imported = next(o for o in addon.bpy.data.objects if o.name == "A tree_trunk_LOD0")
    assert imported.custom_properties["blendkit_asset_id"] == TREE["id"]
    assert imported.custom_properties["blendkit_author"] == "Toby Noby"
    assert imported.custom_properties["blendkit_license"] == "royalty_free"
    assert TREE["id"] in imported.custom_properties["blendkit_url"]


# --- resolutions -------------------------------------------------------------

def test_the_largest_resolution_is_the_default(server, monkeypatch):
    """2K is the top of this asset's ladder, and asking for 4K or 8K would be an
    error naming 0.5K, 1K and 2K."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"])

    assert result["resolution"] == "resolution_2K"
    assert f"downloads/{'33333333-3333-3333-3333-333333333333'}/" in _resolve_calls(calls)[0]["url"]


def test_a_requested_resolution_is_the_one_fetched(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"], resolution="0.5k")

    assert result["resolution"] == "resolution_0_5K"
    assert "downloads/11111111-1111-1111-1111-111111111111/" in _resolve_calls(calls)[0]["url"]


def test_an_unavailable_resolution_is_an_error_naming_the_ones_that_exist(server, monkeypatch):
    """A silent fallback would import 2K when 8K was asked for, and the caller
    would only find out from a render that looked wrong."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"], resolution="8k")

    assert "error" in result
    assert "resolution_0_5K" in result["error"]
    assert "resolution_2K" in result["error"]
    assert _resolve_calls(calls) == []


def test_an_asset_with_no_resolutions_falls_back_to_the_original(server, monkeypatch):
    """Not every asset publishes a resolution ladder; the original .blend is
    then the only file there is."""
    addon, srv = server
    page = {
        "count": 1,
        "results": [{**TREE, "files": [_file("blend", BLEND_UUID, size=900000)]}],
    }
    _install_requests(monkeypatch, addon, page=page, detail=[
        {**TREE, "files": [_file("blend", BLEND_UUID, size=900000)]}])

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert result["resolution"] == "blend"


def test_an_asset_with_no_resolutions_says_so_rather_than_guessing(server, monkeypatch):
    addon, srv = server
    page = {
        "count": 1,
        "results": [{**TREE, "files": [_file("blend", BLEND_UUID)]}],
    }
    _install_requests(monkeypatch, addon, page=page, detail=[
        {**TREE, "files": [_file("blend", BLEND_UUID)]}])

    result = srv.download_blendkit_asset(TREE["id"], resolution="4k")

    assert "error" in result
    assert "publishes no texture resolutions" in result["error"]


def test_the_download_size_is_reported(server, monkeypatch):
    """The single number that decides whether an asset fits. Left to itself the
    caller gets the largest published resolution (11MB here, against 740KB at
    0.5K and 80MB for the original), so the message says what was transferred."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"])

    assert result["resolution"] == "resolution_2K"
    assert "11MB" in result["message"]
    assert "resolution_2K" in result["message"]


def test_a_lower_resolution_reports_its_own_size(server, monkeypatch):
    """740KB rather than 11MB, and the message names which file was taken."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    result = srv.download_blendkit_asset(TREE["id"], resolution="0.5k")

    assert result["resolution"] == "resolution_0_5K"
    assert "resolution_0_5K" in result["message"]


def test_a_format_other_than_blend_is_refused_before_any_transfer(server, monkeypatch):
    """glTF is generated from the .blend, so downloading it when not asked for is
    a large lossy conversion of an asset whose real file is right there."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    for fmt in ("gltf", "fbx", "usd"):
        result = srv.download_blendkit_asset(TREE["id"], file_format=fmt)
        assert "error" in result, fmt
        assert fmt in result["error"]
        assert calls == [], f"{fmt} reached the network before being rejected"


# --- version gating ----------------------------------------------------------

def test_a_newer_blend_is_refused_with_the_versions_named(server, monkeypatch):
    """Blender cannot open a file written by a newer version than itself, and
    Blendkit's assets were saved by everything from 2.93 to 5.0."""
    addon, srv = server
    page = {"count": 1, "results": [{**TREE, "sourceAppVersion": "5.0.0"}]}
    _install_requests(monkeypatch, addon, page=page,
                      detail=[{**TREE, "sourceAppVersion": "5.0.0"}])
    calls = _LAST_CALLS[0]

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "5.0" in result["error"]
    assert "4.2" in result["error"]
    # Refused from the metadata, so the transfer never happened: this .blend is
    # 80MB and there is no point paying for it to find out it cannot be read.
    assert _downloaded(calls) == []
    assert _resolve_calls(calls) == []


def test_a_blend_lying_about_its_version_still_falls_back_to_gltf(monkeypatch, tmp_path):
    """The metadata is only a claim about what wrote the file, so the .blend's
    own header is checked too - and when the two disagree the header wins.

    The fixture's resolve hands back real Blendkit bytes, so the header read here
    is the real one.
    """
    addon = _load_addon(monkeypatch, blender_version=(4, 2, 0))
    monkeypatch.setattr(addon.tempfile, "gettempdir", lambda: str(tmp_path))
    addon._blendkit_scene_uuid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    srv = addon.BlenderMCPServer()

    # Metadata says 4.1 (importable); the bytes say 5.0. Only the .blend gets
    # those bytes - the glTF fallback has to download real-looking GLB magic, or
    # the header parser reads it as a .blend too and refuses it again.
    _install_requests(monkeypatch, addon, page={"count": 1, "results": [TREE]},
                      file_bytes=b"BLENDER17-01v0500" + b"\x00" * 64)

    real_get = addon.requests.get

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        if stream and "glb" in url:
            return FakeResponse(content=b"glTF" + b"\x00" * 32, streamed=True)
        return real_get(url, headers=headers, params=params,
                        timeout=timeout, stream=stream)

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)

    imported = []
    monkeypatch.setattr(
        addon.bpy.ops.import_scene, "gltf",
        lambda filepath=None, **_k: (
            imported.append(filepath),
            addon.bpy.data.objects.append(FakeObject("from_gltf"))))

    result = srv.download_blendkit_asset(TREE["id"])

    # The glTF fallback resolves the *gltf* file, so the fixture has to answer
    # with a gltf fileType for it - which is what the real API does.
    assert result.get("success"), result
    assert len(imported) == 1
    assert imported[0].endswith(".glb")
    assert "glTF" in result["message"]


def test_a_newer_blend_is_refused_before_the_append(server, monkeypatch):
    """The check happens before bpy touches anything, so no half-imported
    datablocks are left behind."""
    addon, srv = server
    page = {"count": 1, "results": [{**TREE, "sourceAppVersion": "5.0.0"}]}
    _install_requests(monkeypatch, addon, page=page,
                      detail=[{**TREE, "sourceAppVersion": "5.0.0"}])
    addon.bpy.data.libraries.contents = BLEND_CONTENTS

    assert "error" in srv.download_blendkit_asset(TREE["id"])
    assert list(addon.bpy.data.objects) == []
    assert list(addon.bpy.context.scene.collection.children) == []


# --- assets that are not geometry --------------------------------------------

def test_a_material_asset_appends_its_material(server, monkeypatch):
    """A material .blend holds one material and no collections or objects, so
    appending objects brought in nothing - and it still reported success, which
    is how a material download 'succeeded' having added no material at all."""
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[
        {**TREE, "assetType": "material",
         "files": [_file("blend", BLEND_UUID, size=235020)]}])
    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": [], "materials": ["Wood Floor White Oak"],
        "node_groups": [],
    }

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert result["materials"] == ["Wood Floor White Oak"]
    assert "Wood Floor White Oak" in result["message"]
    assert result["imported_objects"] == []
    # And it is genuinely in the blend, tagged.
    assert addon.bpy.data.materials.get("Wood Floor White Oak")["blendkit_asset_id"] == TREE["id"]


def test_a_node_group_asset_appends_its_groups(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[
        {**TREE, "assetType": "nodegroup",
         "files": [_file("blend", BLEND_UUID, size=185496)]}])
    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": [], "materials": [],
        "node_groups": ["Voxel Skin"],
    }

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert result["imported_node_groups"] == ["Voxel Skin"]
    assert "Node group(s) added: 1" in result["message"]


def test_a_material_import_packs_its_own_images(server, monkeypatch):
    """There are no objects to reach the images through, so packing has to walk
    the appended material itself - otherwise its textures point at a temporary
    directory that is about to be deleted."""
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[
        {**TREE, "assetType": "material", "files": [_file("blend", BLEND_UUID)]}])

    image = FakeImage("floor_color.jpg")
    material = FakeMaterial("Oak")
    material.use_nodes = True
    material.node_tree = types.SimpleNamespace(
        nodes=[types.SimpleNamespace(type="TEX_IMAGE", image=image)], links=[])

    addon.bpy.data.materials.new = lambda name: material
    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": [], "materials": ["Oak"], "node_groups": []}

    assert srv.download_blendkit_asset(TREE["id"]).get("success")
    assert image.packed_file is not None


def test_an_empty_blend_is_reported_as_a_failure_not_a_success(server, monkeypatch):
    """Nothing appended is a broken import, and saying otherwise hides it."""
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[
        {**TREE, "assetType": "material", "files": [_file("blend", BLEND_UUID)]}])
    addon.bpy.data.libraries.contents = {
        "collections": {}, "objects": [], "materials": [], "node_groups": []}

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "nothing this integration can append" in result["error"]


def test_a_collection_with_no_name_match_is_still_appended(server, monkeypatch):
    """Better than importing nothing, and the response says what arrived."""
    addon, srv = server
    _install_requests(monkeypatch, addon)
    addon.bpy.data.libraries.contents = {
        "collections": {"Some Other Name": ["thing"]},
        "objects": ["thing"], "materials": [], "node_groups": [],
    }

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert "Some Other Name" in result["imported_collections"]


# --- summaries ---------------------------------------------------------------

def test_the_summary_carries_the_numbers_that_decide_fit(server, monkeypatch):
    """A 1.4m-face tree and a 400-face one are both named "a tree". The face
    count, the real-world size and the download size are what separate them, and
    none of them are visible in the thumbnail."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert asset["face_count"] == 1439949
    assert asset["dimension_z"] == pytest.approx(14.1296)
    assert asset["download_size"] == 83469545
    assert asset["resolutions"] == ["resolution_0_5K", "resolution_1K", "resolution_2K"]
    assert asset["license"] == "royalty_free"
    assert asset["author"] == "Toby Noby"
    assert asset["blender_version"] == (4, 1)


def test_a_gltf_only_asset_is_flagged_as_importable_but_lossy(server, monkeypatch):
    """A handful of assets publish no .blend. They are still worth offering -
    glTF beats no model - but the caller is told the materials are converted."""
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE, "files": [_file("gltf", GLTF_UUID, size=15853120)],
    }]}
    _install_requests(monkeypatch, addon, page=page)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert asset["importable"] is True
    assert "conversion" in asset["import_note"]


def test_a_zip_only_asset_is_flagged_unimportable(server, monkeypatch):
    """Blendkit publishes some assets as a zip archive and nothing else. Saying so
    up front is better than a download that fails."""
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE, "files": [_file("zip_file", "77777777-7777-7777-7777-777777777777")],
    }]}
    _install_requests(monkeypatch, addon, page=page)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert asset["importable"] is False
    assert "zip" in asset["import_note"]


# --- HDR assets are images, not blends ---------------------------------------

def test_the_suffix_comes_from_the_filename_not_the_file_type(server):
    """An HDR asset's files all report fileType "blend" while shipping OpenEXR.
    Trusting the fileType sent every HDR to bpy.data.libraries.load, which
    cannot read an .exr and silently appends nothing."""
    addon, srv = server
    assert addon._blendkit_file_suffix({"fileType": "blend", "filename": "a/blend_x.exr"}) == ".exr"
    assert addon._blendkit_file_suffix({"fileType": "blend", "filename": "a/blend_x.BLEND"}) == ".blend"
    assert addon._blendkit_file_suffix({"fileType": "blend"}) == ""


HDR_ASSET = {
    **TREE,
    "assetType": "hdr",
    "files": [
        # fileType lies on purpose: this is what the live API returns.
        {**_file("resolution_0_5K", "aaaa1111-2222-3333-4444-555566667777", size=70741),
         "filename": "assets/x/files/resolution_0_5K_1.exr"},
        {**_file("blend", "bbbb1111-2222-3333-4444-555566667777", size=419296055),
         "filename": "assets/x/files/blend_1.exr"},
    ],
}


def test_an_hdr_asset_becomes_a_world_not_an_append(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[HDR_ASSET],
                      page={"count": 1, "results": [HDR_ASSET]})

    result = srv.download_blendkit_asset(TREE["id"], resolution="0.5k")

    assert result.get("success"), result
    assert result["world"] == f"Blendkit {HDR_ASSET['name']}"
    world = addon.bpy.data.worlds.get(result["world"])
    assert world is not None
    # Packed, or the world loses its lighting once the temp file is gone.
    image = addon.bpy.data.images.get(result["image_name"])
    assert image is not None and image.packed_file is not None
    assert addon.bpy.context.scene.world is world
    assert world.use_nodes is True
    kinds = {n.type for n in world.node_tree.nodes}
    assert "TEX_ENVIRONMENT" in kinds and "BACKGROUND" in kinds
    # No geometry was expected, and none was invented.
    assert result.get("imported_objects", []) == []


def test_an_hdr_import_does_not_wipe_an_existing_world(server, monkeypatch):
    """The world already in the scene may be hand-built. A new one is made and
    activated instead of clearing the node tree of whatever was there."""
    addon, srv = server
    theirs = addon.bpy.data.worlds.new("Hand Built")
    theirs.node_tree.nodes.new(type="ShaderNodeBackground")
    addon.bpy.context.scene.world = theirs
    _install_requests(monkeypatch, addon, detail=[HDR_ASSET],
                      page={"count": 1, "results": [HDR_ASSET]})

    result = srv.download_blendkit_asset(TREE["id"], resolution="0.5k")

    assert result.get("success"), result
    assert len(theirs.node_tree.nodes) == 1, "an unrelated world was wiped"
    assert addon.bpy.context.scene.world is not theirs


def test_an_hdr_summary_says_it_becomes_a_world(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[HDR_ASSET],
                      page={"count": 1, "results": [HDR_ASSET]})

    asset = srv.search_blendkit_assets("sunset", asset_type="hdr")["assets"][0]

    assert asset["import_kind"] == "world"


def test_a_model_summary_still_says_append(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert asset["import_kind"] == "append"


def test_download_looks_the_asset_up_by_the_id_search_hands_out(server, monkeypatch):
    """Search reports `id` (the version uuid) and download is handed that same
    value, so download has to accept it.

    It used to look the asset up through search with `asset_base_id:<id>`, which
    is a *different* uuid - and the version id is not a searchable field at all,
    so every download of a freshly searched asset came back "no such asset".
    Only the detail endpoint `/assets/<version id>/` serves it anonymously.
    """
    addon, srv = server
    _install_requests(monkeypatch, addon)
    calls = _LAST_CALLS[0]

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    detail = [c for c in calls if "/api/v1/assets/" in c["url"]]
    assert len(detail) == 1
    assert detail[0]["url"].endswith(f"/assets/{TREE['id']}/")
    # And no search round-trip at all: the detail record already has the files.
    assert _search_calls(calls) == []


def test_download_does_not_require_can_download(server, monkeypatch):
    """The detail endpoint omits `canDownload` - it is a per-user entitlement the
    search endpoint computes and this one does not. Gating on it here rejected
    every free asset with a "paid" message."""
    addon, srv = server
    _install_requests(monkeypatch, addon, detail=[{k: v for k, v in TREE.items()
                                                   if k != "canDownload"}])

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert "Full Plan" not in result.get("message", "")


def test_a_zip_only_asset_is_refused_with_instructions_not_a_crash(server, monkeypatch):
    """Its layout is chosen by the uploader, so unpacking one blind is how a
    remote API ends up writing outside the temp directory."""
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE, "files": [_file("zip_file", "77777777-7777-7777-7777-777777777777")],
    }]}
    _install_requests(monkeypatch, addon, page=page, detail=[
        {**TREE, "files": [_file("zip_file", "77777777-7777-7777-7777-777777777777")]}])
    calls = _LAST_CALLS[0]

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "zip archive" in result["error"]
    assert "File > Import" in result["error"]
    assert _downloaded(calls) == []


def test_an_asset_with_no_usable_file_names_what_it_does_have(server, monkeypatch):
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE, "files": [_file("thumbnail", "88888888-8888-8888-8888-888888888888"),
                         _file("prxc", "99999999-9999-9999-9999-999999999999")],
    }]}
    _install_requests(monkeypatch, addon, page=page, detail=[
        {**TREE, "files": [_file("thumbnail", "88888888-8888-8888-8888-888888888888"),
                          _file("prxc", "99999999-9999-9999-9999-999999999999")]}])

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "no importable file" in result["error"]
    assert "prxc" in result["error"]


def test_a_gltf_only_asset_downloads_and_imports(server, monkeypatch):
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE, "files": [_file("gltf", GLTF_UUID, size=15853120)],
    }]}
    _install_requests(monkeypatch, addon, page=page, detail=[
        {**TREE, "files": [_file("gltf", GLTF_UUID, size=15853120)]}])

    # The glTF importer really creates objects; the stand-in has to as well, or
    # the import legitimately looks like it brought nothing in.
    imported = []
    monkeypatch.setattr(
        addon.bpy.ops.import_scene, "gltf",
        lambda filepath=None, **_k: (
            imported.append(filepath),
            addon.bpy.data.objects.append(FakeObject("from_gltf"))))

    result = srv.download_blendkit_asset(TREE["id"])

    assert result.get("success"), result
    assert result["resolution"] == "gltf"
    assert len(imported) == 1 and imported[0].endswith(".glb")
    assert result["imported_objects"] == ["from_gltf"]
    assert "glTF" in result["message"]


def test_the_summary_drops_the_bulk_of_the_record(server, monkeypatch):
    """The raw record is ~4KB per asset - social links, per-avatar URL variants,
    marketing label plumbing - so 20 results is 80KB of mostly noise."""
    addon, srv = server
    _install_requests(monkeypatch, addon)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert set(asset) <= {
        "id", "asset_base_id", "name", "asset_type", "description", "author",
        "tags", "category", "license", "access", "access_label", "can_download",
        "is_free", "verification_status", "source_app_version", "resolutions",
        "url", "thumbnail", "download_size", "gltf_download_size", "face_count",
        "texture_resolution", "dimension_x", "dimension_y", "dimension_z",
        "object_count", "blender_version", "has_gltf", "importable", "import_note",
        "import_kind",
    }
    assert "socialNetworks" not in asset


def test_resolutions_are_listed_smallest_first(server, monkeypatch):
    """So "the biggest available" is a comparison rather than a guess."""
    addon, srv = server
    page = {"count": 1, "results": [{
        **TREE,
        "files": [
            _file("resolution_8K", "8"),
            _file("resolution_0_5K", "05"),
            _file("resolution_4K", "4"),
            _file("resolution_1K", "1"),
        ],
    }]}
    _install_requests(monkeypatch, addon, page=page)

    asset = srv.search_blendkit_assets("tree")["assets"][0]

    assert asset["resolutions"] == [
        "resolution_0_5K", "resolution_1K", "resolution_4K", "resolution_8K"]


# --- previews ----------------------------------------------------------------

def test_a_preview_needs_no_second_search(server, monkeypatch):
    """Blendkit has no per-asset record endpoint, so the thumbnail URL comes from
    the search result the caller already has."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.get_blendkit_asset_preview(TREE["id"], thumbnail_url=TREE["thumbnailSmallUrl"])

    assert result.get("success"), result
    assert _search_calls(calls) == []
    assert result["format"] == "jpeg"


def test_the_preview_thumbnail_is_resized_from_the_published_url(server, monkeypatch):
    """The CDN derives the size from the filename suffix, so a preview costs no
    stored file and no extra request - and a search result's 1024px URL is
    downsized to the preview size rather than transferred at full size."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.get_blendkit_asset_preview(TREE["id"], thumbnail_url=TREE["thumbnailLargeUrl"])

    assert _downloaded(calls) == []
    url = calls[0]["url"]
    assert ".256x256_q85_crop-,.jpg" in url
    assert "1024x1024" not in url


def test_a_preview_size_the_cdn_does_not_serve_falls_back_to_a_published_one(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.get_blendkit_asset_preview(TREE["id"], thumbnail_url=TREE["thumbnailLargeUrl"],
                                   size=777)

    # Not a size the CDN serves, so the smallest published one is used rather
    # than a URL that 404s.
    assert ".256x256_q85_crop-" in calls[0]["url"]


def test_an_unfamiliar_thumbnail_url_is_used_as_given(server, monkeypatch):
    """A shape this code does not recognise must still produce a working preview
    rather than a mangled URL."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)
    odd = "https://public.blenderkit.com/thumbnails/assets/x/files/y.jpg"

    result = srv.get_blendkit_asset_preview(TREE["id"], thumbnail_url=odd)

    assert result.get("success"), result
    assert calls[0]["url"] == odd


def test_a_preview_without_a_url_says_so(server, monkeypatch):
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    result = srv.get_blendkit_asset_preview(TREE["id"])

    assert "error" in result
    assert "thumbnail_url" in result["error"]
    assert calls == []


def test_an_asset_id_cannot_escape_a_path(server, monkeypatch):
    """asset_id arrives from the model and is interpolated into URLs."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    for hostile in ("../../etc/passwd", "a/../../b", "x?y=1", "x y"):
        result = srv.download_blendkit_asset(hostile)
        assert "error" in result, hostile
        result = srv.get_blendkit_asset_preview(hostile, thumbnail_url=THUMB)
        assert "error" in result, hostile

    assert _search_calls(calls) == []
    # Real ids still pass.
    assert addon._blendkit_valid_asset_id(TREE["id"])
    assert addon._blendkit_valid_asset_id("0a00681c-598c-4225-9f67-b69e6642f5dc")


# --- errors the API actually returns -----------------------------------------

def test_a_401_on_download_is_reported_as_a_plan_problem(server, monkeypatch):
    """What a Full Plan asset answers, even though search offered a downloadUrl."""
    addon, srv = server

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        if "/downloads/" in url:
            return FakeResponse(
                status_code=401,
                payload={"detail": "Unauthorized access. Please log in.", "statusCode": 401},
            )
        if "/api/v1/assets/" in url:
            return FakeResponse(payload={k: v for k, v in TREE.items()
                                         if k != "canDownload"})
        return FakeResponse(payload=PAGE)

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "Full Plan" in result["error"]
    # Not "check your API key" - there isn't one.
    assert "API key" not in result["error"]


def test_a_missing_scene_uuid_is_reported_as_our_bug(server, monkeypatch):
    """The 403 body is "Parameter scene_uuid not set", which reads exactly like a
    permissions problem. Telling the user to log in would send them nowhere."""
    addon, srv = server

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        if "/downloads/" in url:
            return FakeResponse(
                status_code=403,
                payload={"detail": "Parameter scene_uuid not set", "statusCode": 403},
            )
        if "/api/v1/assets/" in url:
            return FakeResponse(payload={k: v for k, v in TREE.items()
                                         if k != "canDownload"})
        return FakeResponse(payload=PAGE)

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "bug" in result["error"]
    assert "log in" not in result["error"]


def test_a_404_names_the_asset_as_gone(server, monkeypatch):
    addon, srv = server

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        if "/downloads/" in url:
            return FakeResponse(status_code=404, payload={"detail": "Not found."})
        if "/api/v1/assets/" in url:
            return FakeResponse(payload={k: v for k, v in TREE.items()
                                         if k != "canDownload"})
        return FakeResponse(payload=PAGE)

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "search again" in result["error"]


def _detail_payload(*assets):
    """The asset detail record, which unlike a search record omits canDownload."""
    return next(iter(assets)) if len(assets) == 1 else {}


def _detail_ok(*assets):
    """A requests.get stand-in that answers the detail endpoint and delegates
    the rest to the full router, for the tests that only override one status."""
    base = _install_requests

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        if "/api/v1/assets/" in url:
            ident = url.split("/api/v1/assets/")[-1].split("/")[0]
            for candidate in assets:
                if candidate["id"] == ident:
                    return FakeResponse(payload={
                        k: v for k, v in candidate.items() if k != "canDownload"})
            return FakeResponse(status_code=404, payload={"detail": "Not found."})
        return base(url, headers=headers, params=params, timeout=timeout, stream=stream)

    return fake_get


def test_a_rate_limited_search_says_to_wait(server, monkeypatch):
    addon, srv = server

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        return FakeResponse(status_code=429, payload={"detail": "Too many requests."})

    monkeypatch.setattr(addon.requests, "get", fake_get, raising=False)

    result = srv.search_blendkit_assets("tree")

    assert "rate limiting" in result["error"]


def test_an_unknown_id_is_reported_without_a_transfer(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon, page={"count": 0, "results": []})

    result = srv.download_blendkit_asset("603d2a33-426b-4690-8982-13520b58d000")

    assert "No Blendkit asset found" in result["error"]


def test_a_malformed_file_reference_is_refused(server, monkeypatch):
    """The file uuid arrives from the API and is interpolated into the resolve
    URL, so it is checked before it is used."""
    addon, srv = server
    bad = {**TREE, "files": [{**_file("blend", BLEND_UUID), "uuid": "../../evil"}]}
    _install_requests(monkeypatch, addon, page={"count": 1, "results": [bad]}, detail=[bad])

    result = srv.download_blendkit_asset(TREE["id"])

    assert "error" in result
    assert "malformed" in result["error"]


def test_a_resolve_returning_the_wrong_file_type_is_refused(server, monkeypatch):
    """A resolution_* file is a .blend under a different name. Resolving one and
    being handed a glTF means the response cannot be trusted to match what was
    asked for, so it is refused rather than imported as the wrong thing."""
    addon, srv = server
    _install_requests(monkeypatch, addon, resolve={
        "fileType": "gltf", "uuid": "33333333-3333-3333-3333-333333333333",
        "filePath": "https://assets.blenderkit.com/x.glb?verify=1-hmac",
    })

    result = srv.download_blendkit_asset(TREE["id"], resolution="2k")

    assert "error" in result
    assert "where a resolution_2K was asked for" in result["error"]


def test_a_resolve_with_no_file_path_is_refused(server, monkeypatch):
    addon, srv = server
    _install_requests(monkeypatch, addon, resolve={
        "fileType": "resolution_2K", "uuid": "33333333-3333-3333-3333-333333333333",
    })

    result = srv.download_blendkit_asset(TREE["id"], resolution="2k")

    assert "error" in result
    assert "no file path" in result["error"]


# --- status ------------------------------------------------------------------

def test_the_status_needs_no_api_key(server):
    """Blendkit is the only integration here with nothing to configure, so the
    status message says so rather than sending the user looking for a key."""
    addon, srv = server

    enabled = srv.get_blendkit_status()
    assert enabled["enabled"] is True
    assert "No API key" in enabled["message"]

    addon.bpy.context.scene.blendermcp_use_blendkit = False
    disabled = srv.get_blendkit_status()
    assert disabled["enabled"] is False
    assert "Use assets from Blendkit" in disabled["message"]


def test_requests_identify_the_integration(server, monkeypatch):
    """Versioned in the User-Agent, so Blendkit can tell which integration its
    traffic is and how many people it is serving."""
    addon, srv = server
    calls = _install_requests(monkeypatch, addon)

    srv.search_blendkit_assets("tree")

    agent = calls[0]["headers"]["User-Agent"]
    assert agent.startswith("blender-mcp/1.8")
