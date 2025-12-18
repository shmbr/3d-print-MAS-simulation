import React, { Suspense } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { STLLoader } from "three-stdlib";

function Model({ url }: { url: string }) {
  const geometry = useLoader(STLLoader, url);
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial color="orange" />
    </mesh>
  );
}

export default function StlViewer({ url }: { url: string }) {
  return (
    <div className="h-[500px] w-full rounded-md overflow-hidden border">
      <Canvas camera={{ position: [0, 0, 70], fov: 100 }}>
        <ambientLight />
        <directionalLight position={[10, 10, 10]} />
        <Suspense fallback={null}>
          <Model url={url} />
        </Suspense>
        <OrbitControls />
      </Canvas>
    </div>
  );
}
