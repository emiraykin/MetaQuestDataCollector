import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { PoseFrame, Region, Status } from "../protocol";
import { MujocoScene } from "../MujocoScene";

// MuJoCo z-up world -> three.js y-up (same basis change the headset uses, without room placement).
const Z_UP_TO_Y_UP = new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, 0, -1), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0),
);
const SEND_MS = 33;

/** Scene-editor mode of the viewer: click an object to select it, drag the gizmo to move (W) or turn (E) it. */
export interface ViewerEditor {
  objects: string[];                 // movable task objects
  selected: string | null;
  onSelect: (name: string | null) => void;
  onMove: (name: string, pos: [number, number], yaw: number) => void;  // world x, y (m) and yaw (rad)
  regions: Record<string, Region>;
  spawn: Record<string, Region>;
  colors: Record<string, number[]>;  // object -> rgba, for its start region outline
}

export interface ViewerHandle {
  push: (f: PoseFrame) => void;
  reload: () => void;
  status: (s: Status) => void;
  setEditor: (e: ViewerEditor | null) => void;
}

export function Viewer3D({ handle }: { handle: React.MutableRefObject<ViewerHandle | null> }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0a0e18");
    scene.fog = new THREE.Fog("#0a0e18", 4, 12);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 50);
    camera.position.set(-0.2, 1.75, 1.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.85, -0.45);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;

    const mj = new MujocoScene({ shadows: true });
    mj.setPlacement(Z_UP_TO_Y_UP.toArray());
    scene.add(mj.root);  // App calls reload() once the websocket connects

    // MuJoCo world axes at the origin: red = +x, green = +y, blue = +z (up). Same z-up -> y-up basis as the scene;
    // drawn on top so the table doesn't hide them.
    const axes = new THREE.AxesHelper(0.5);
    axes.matrixAutoUpdate = false;
    axes.matrix.copy(Z_UP_TO_Y_UP);
    (axes.material as THREE.Material).depthTest = false;
    axes.renderOrder = 999;
    scene.add(axes);

    // ---- editor: a proxy object follows the selected body; the gizmo moves the proxy, the proxy sends moves ----
    let editor: ViewerEditor | null = null;
    const proxy = new THREE.Object3D();
    scene.add(proxy);
    const gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setSpace("world");
    gizmo.setSize(0.8);
    const helper = gizmo.getHelper();
    scene.add(helper);
    const setMode = (m: "translate" | "rotate") => {
      gizmo.setMode(m);
      gizmo.showX = gizmo.showZ = m === "translate";  // the table plane (three x/z = MuJoCo y/x)
      gizmo.showY = m === "rotate";                    // yaw only (three y = MuJoCo z)
    };
    setMode("translate");
    let dragging = false, justDragged = false, lastSend = 0;
    const rootInv = new THREE.Matrix4(), local = new THREE.Matrix4(), rootQ = new THREE.Quaternion();
    const zAxis = new THREE.Vector3(0, 0, 1), tmpQ = new THREE.Quaternion();
    /** The proxy's pose in MuJoCo world coordinates: x, y and yaw. */
    const proxyPose = (): [[number, number], number] => {
      rootInv.copy(mj.root.matrixWorld).invert();
      local.multiplyMatrices(rootInv, proxy.matrixWorld);
      const e = local.elements;
      return [[e[12], e[13]], Math.atan2(e[1], e[0])];
    };
    const send = (force = false) => {
      if (!editor?.selected) return;
      const now = performance.now();
      if (!force && now - lastSend < SEND_MS) return;
      lastSend = now;
      const [pos, yaw] = proxyPose();
      editor.onMove(editor.selected, pos, yaw);
    };
    gizmo.addEventListener("dragging-changed", (ev) => {
      dragging = !!ev.value;
      controls.enabled = !dragging;
      if (dragging) justDragged = true;
      else send(true);
    });
    gizmo.addEventListener("objectChange", () => send());
    /** Put the proxy on the selected body: its position and only its yaw (the gizmo turns about the vertical). */
    const syncProxy = (body: THREE.Object3D) => {
      body.updateWorldMatrix(true, false);
      body.getWorldPosition(proxy.position);
      rootInv.copy(mj.root.matrixWorld).invert();
      local.multiplyMatrices(rootInv, body.matrixWorld);
      const e = local.elements;
      mj.root.getWorldQuaternion(rootQ);
      proxy.quaternion.copy(rootQ).multiply(tmpQ.setFromAxisAngle(zAxis, Math.atan2(e[1], e[0])));
      proxy.updateMatrixWorld();
    };
    const applyEditor = () => {
      const sel = editor?.selected ?? null;
      mj.highlightBody(null);
      mj.highlightBody(sel);
      if (editor) mj.setOverlays(editor.regions, editor.spawn, editor.colors);
      else mj.setOverlays(null);
      const body = sel ? mj.bodyObject(sel) : null;
      if (body) { syncProxy(body); gizmo.attach(proxy); } else gizmo.detach();
    };

    // Click (not drag) selects the object under the pointer, or nothing.
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; justDragged = false; };
    const onUp = (e: PointerEvent) => {
      const d = down;
      down = null;
      if (!editor || !d || justDragged || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
      const r = renderer.domElement.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = mj.pickObject(ray, editor.objects);
      editor.onSelect(hit?.name ?? null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (!editor || (e.target as HTMLElement)?.closest?.("input, textarea, select")) return;
      if (e.key === "w" || e.key === "W") setMode("translate");
      else if (e.key === "e" || e.key === "E") setMode("rotate");
      else if (e.key === "Escape") editor.onSelect(null);
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    addEventListener("keydown", onKey);

    handle.current = {
      push: (f) => { if (!f.ghost) mj.pushFrame(f, performance.now()); },
      reload: () => { void mj.load().then(applyEditor); },
      status: (s) => mj.setOutOfReach(s.out_of_reach),
      setEditor: (e) => { editor = e; applyEditor(); },
    };

    // Debug handle for automated tests and the browser console (like window.__xrc on the headset page).
    (window as unknown as { __xrcViewer: unknown }).__xrcViewer = { renderer, camera, mj, proxy, gizmo };

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    renderer.setAnimationLoop(() => {
      controls.update();
      mj.update(performance.now());
      if (editor?.selected && !dragging) {
        const body = mj.bodyObject(editor.selected);
        if (body) syncProxy(body);
      }
      helper.visible = !!editor?.selected && !!mj.bodyObject(editor.selected);
      renderer.render(scene, camera);
    });
    return () => {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      removeEventListener("keydown", onKey);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      gizmo.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      handle.current = null;
    };
  }, [handle]);

  return (
    <>
      <div className="viewer" ref={host} />
      <div className="axes-legend">
        <span><i style={{ background: "#ff0000" }} />x · away</span>
        <span><i style={{ background: "#00ff00" }} />y · left</span>
        <span><i style={{ background: "#0000ff" }} />z · up</span>
      </div>
    </>
  );
}
