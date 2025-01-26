'use client'

import dynamic from 'next/dynamic'

const CameraView = dynamic(
  () => import('@/components/CameraView').then((mod) => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="fixed inset-0 flex items-center justify-center bg-black">
        <p className="text-xl text-white">Memuat kamera...</p>
      </div>
    ),
  }
)

export default function Home() {
  return (
    <div suppressHydrationWarning>
      <CameraView />
    </div>
  )
}
