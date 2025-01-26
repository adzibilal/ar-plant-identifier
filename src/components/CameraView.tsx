'use client'

import { useEffect, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as cocossd from '@tensorflow-models/coco-ssd'
import { analyzePlantImage } from '@/utils/openai'
import { Camera, X } from 'lucide-react'
import toast, { Toaster } from 'react-hot-toast'

export default function CameraView() {
  const webcamRef = useRef<Webcam>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null)
  const [detectorModel, setDetectorModel] = useState<cocossd.ObjectDetection | null>(null)
  const [prediction, setPrediction] = useState<string>('')
  const [plantInfo, setPlantInfo] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 })
  const [showModal, setShowModal] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<{
    isPlant: boolean
    name?: string
    description?: string
    confidence?: number
    error?: string
    details?: {
      scientificName?: string
      family?: string
      type?: string
      habitat?: string
      characteristics?: {
        height?: string
        color?: string
        leafType?: string
        flowerType?: string
      }
      uses?: string[]
      careInstructions?: {
        watering?: string
        sunlight?: string
        soil?: string
        temperature?: string
      }
    }
  } | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)

  // Daftar kata kunci yang lebih komprehensif untuk deteksi tanaman
  const plantKeywords = [
    // Kategori umum
    'potted plant', 'plant', 'tree', 'flower', 'grass',
    'leaf', 'palm tree', 'bush', 'vine', 'herb',
    'garden', 'forest', 'vegetation', 'flora', 'branch',
    
    // Tanaman Spesifik
    'rose', 'lily', 'orchid', 'cactus', 'bamboo',
    'fern', 'moss', 'algae', 'succulent', 'bonsai',
    'sunflower', 'tulip', 'daisy', 'maple', 'oak',
    'pine', 'palm', 'banana', 'mango', 'apple',
    
    // Bagian Tanaman
    'stem', 'trunk', 'root', 'branch', 'twig',
    'bud', 'shoot', 'seedling', 'sapling', 'sprout',
    
    // Habitat
    'greenhouse', 'garden', 'forest', 'jungle', 'park',
    
    // Deskripsi
    'green', 'leafy', 'flowering', 'tropical', 'botanical',
    'evergreen', 'deciduous', 'perennial', 'annual'
  ]

  // Fungsi yang lebih sederhana dan lebih sensitif untuk deteksi tanaman
  const isPlant = (className: string, confidence: number): boolean => {
    const normalizedClass = className.toLowerCase()
    
    // Daftar kata kunci utama dengan threshold rendah
    const primaryKeywords = [
      'plant', 'tree', 'flower', 'leaf', 'grass',
      'potted', 'garden', 'bush', 'vine'
    ]

    // Cek kata kunci utama dengan threshold yang sangat rendah
    const hasPrimaryKeyword = primaryKeywords.some(keyword => 
      normalizedClass.includes(keyword)
    )

    // Terima semua objek yang terdeteksi sebagai tanaman dengan confidence > 0.2
    if (hasPrimaryKeyword && confidence > 0.2) {
      return true
    }

    // Cek objek umum yang mungkin tanaman
    const generalObjects = [
      'object', 'thing', 'item', 'green', 'nature',
      'outdoor', 'garden', 'park'
    ]

    // Terima objek umum dengan confidence yang lebih tinggi
    if (generalObjects.some(obj => normalizedClass.includes(obj)) && confidence > 0.4) {
      return true
    }

    return false
  }

  useEffect(() => {
    setDimensions({
      width: window.innerWidth,
      height: window.innerHeight
    })

    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    const initializeTF = async () => {
      try {
        await tf.setBackend('webgl')
        console.log('Using backend:', tf.getBackend())
        
        const [loadedMobilenet, loadedDetector] = await Promise.all([
          mobilenet.load(),
          cocossd.load()
        ])
        
        setModel(loadedMobilenet)
        setDetectorModel(loadedDetector)
        setIsLoading(false)
        console.log('Model berhasil dimuat')
      } catch (error) {
        console.error('Error initializing:', error)
        try {
          await tf.setBackend('cpu')
          console.log('Fallback to CPU backend')
          const [loadedMobilenet, loadedDetector] = await Promise.all([
            mobilenet.load(),
            cocossd.load()
          ])
          setModel(loadedMobilenet)
          setDetectorModel(loadedDetector)
          setIsLoading(false)
        } catch (cpuError) {
          console.error('Error pada CPU fallback:', cpuError)
        }
      }
    }

    initializeTF()
  }, [])

  useEffect(() => {
    let animationFrameId: number

    const detect = async () => {
      if (!detectorModel || !webcamRef.current || !canvasRef.current) return

      if (webcamRef.current.video?.readyState !== 4) {
        animationFrameId = requestAnimationFrame(detect)
        return
      }

      const video = webcamRef.current.video
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      try {
        const predictions = await detectorModel.detect(video)
        
        if (context) {
          context.clearRect(0, 0, canvas.width, canvas.height)
          
          predictions
            .filter(pred => isPlant(pred.class, pred.score))
            .forEach(prediction => {
              const [x, y, width, height] = prediction.bbox
              const confidence = Math.round(prediction.score * 100)

              // Gambar bounding box dengan gradien dan efek yang lebih halus
              const gradient = context.createLinearGradient(x, y, x + width, y + height)
              gradient.addColorStop(0, `rgba(0, 255, 0, ${prediction.score * 0.8})`)
              gradient.addColorStop(1, `rgba(0, 200, 0, ${prediction.score * 0.8})`)
              
              context.shadowColor = 'rgba(0, 255, 0, 0.5)'
              context.shadowBlur = 15
              context.strokeStyle = gradient
              context.lineWidth = 3
              context.strokeRect(x, y, width, height)
              
              // Label yang lebih informatif
              context.shadowBlur = 0
              context.fillStyle = 'rgba(0, 0, 0, 0.8)'
              const label = `${prediction.class} (${confidence}%)`
              const labelWidth = context.measureText(label).width + 20
              const labelHeight = 30
              
              // Background label dengan sudut membulat
              context.beginPath()
              context.roundRect(x, y - labelHeight - 5, labelWidth, labelHeight, 5)
              context.fill()

              // Teks label
              context.fillStyle = '#ffffff'
              context.font = 'bold 16px Arial'
              context.fillText(label, x + 10, y - 15)
            })
        }
      } catch (error) {
        console.error('Error in detection:', error)
      }

      animationFrameId = requestAnimationFrame(detect)
    }

    detect()

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
    }
  }, [detectorModel])

  const cropImageToPlant = async (imageElement: HTMLImageElement, bbox: [number, number, number, number]) => {
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) return null

    const [x, y, width, height] = bbox
    
    // Tambahkan padding 10% di sekitar area tanaman
    const padding = {
      x: width * 0.1,
      y: height * 0.1
    }

    const cropX = Math.max(0, x - padding.x)
    const cropY = Math.max(0, y - padding.y)
    const cropWidth = Math.min(width + (padding.x * 2), imageElement.width - cropX)
    const cropHeight = Math.min(height + (padding.y * 2), imageElement.height - cropY)

    canvas.width = cropWidth
    canvas.height = cropHeight

    context.drawImage(
      imageElement,
      cropX, cropY, cropWidth, cropHeight,
      0, 0, cropWidth, cropHeight
    )

    return canvas.toDataURL('image/jpeg')
  }

  const captureAndAnalyze = async () => {
    if (!webcamRef.current || isAnalyzing) return

    try {
      setIsAnalyzing(true)
      const imageElement = webcamRef.current.getScreenshot()
      
      if (imageElement) {
        setCapturedImage(imageElement)
        
        // Analisis gambar menggunakan OpenAI
        const result = await analyzePlantImage(imageElement)
        setAnalysisResult(result)
        console.debug(result)

        // Tampilkan toast error jika bukan tanaman
        if (!result.isPlant) {
          toast.error(result.error || 'Tidak ada tanaman terdeteksi dalam gambar', {
            duration: 3000,
            position: 'top-center',
            style: {
              background: '#333',
              color: '#fff',
              borderRadius: '12px',
            }
          })
        }
        
        // Selalu tampilkan modal dan preview
        setShowModal(true)
      }
    } catch (error) {
      console.error('Error capturing image:', error)
      toast.error('Gagal mengambil gambar')
    } finally {
      setIsAnalyzing(false)
    }
  }

  if (isLoading) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-black">
        <p className="text-xl text-white">Memuat model AI...</p>
      </div>
    )
  }

  return (
    <main className="fixed inset-0 bg-black flex justify-center">
      <div className="relative w-full max-w-[428px] h-full">
        <div className="relative h-full w-full">
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="h-full w-full object-cover"
            videoConstraints={{
              width: { ideal: dimensions.width },
              height: { ideal: dimensions.height },
              facingMode: 'environment',
              aspectRatio: dimensions.width / dimensions.height
            }}
          />
          
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
          />

          <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/90 via-black/60 to-transparent">
            <div className="px-4 py-3 flex items-center justify-between">
              <h1 className="text-white text-lg font-semibold">
                AR Plant Identifier
              </h1>
              <span className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full">
                Live
              </span>
            </div>
          </div>

          <div className="absolute bottom-0 left-0 right-0">
            <div className="bg-gradient-to-t from-black via-black/80 to-transparent pb-8 pt-16">
              <div className="flex justify-center items-center gap-6">
                <button
                  onClick={captureAndAnalyze}
                  disabled={isAnalyzing}
                  className={`w-18 h-18 rounded-full flex items-center justify-center p-1
                    ${isAnalyzing 
                      ? 'bg-gray-600' 
                      : 'bg-white hover:bg-gray-200 active:bg-gray-300'
                    } transition-all transform active:scale-95 shadow-lg`}
                >
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center
                    ${isAnalyzing ? 'bg-gray-500' : 'bg-white'}`}>
                    {isAnalyzing ? (
                      <div className="w-8 h-8 border-4 border-gray-800 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Camera className="w-8 h-8 text-black" />
                    )}
                  </div>
                </button>
              </div>

              <p className="text-white/60 text-xs text-center mt-4">
                Ketuk untuk menganalisis tanaman
              </p>
            </div>
          </div>

          {showModal && analysisResult && (
            <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 backdrop-blur-sm">
              <div className="w-full mx-4 mb-4 animate-slide-up max-w-[400px] max-h-[90vh] flex flex-col">
                <div className="bg-white rounded-2xl overflow-hidden shadow-xl flex flex-col">
                  {capturedImage && (
                    <div className="mx-auto py-2">
                      <img 
                        src={capturedImage} 
                        alt="Captured Image"
                        className="w-auto h-[200px]"
                      />
                    </div>
                  )}
                  <div className={`px-6 py-4 ${analysisResult.isPlant ? 'bg-green-50' : 'bg-red-50'} flex items-center justify-between shrink-0`}>
                    <div>
                      <h3 className={`text-sm font-medium ${analysisResult.isPlant ? 'text-green-600' : 'text-red-600'}`}>
                        {analysisResult.isPlant ? 'HASIL IDENTIFIKASI' : 'HASIL ANALISIS'} ({analysisResult.confidence}% yakin)
                      </h3>
                      <p className="text-xl font-bold text-gray-900">
                        {analysisResult.isPlant ? analysisResult.name : 'Bukan Tanaman'}
                      </p>
                    </div>
                    <button 
                      onClick={() => {
                        setShowModal(false)
                        setCapturedImage(null)
                        setAnalysisResult(null)
                      }}
                      className="p-2 hover:bg-green-100 rounded-full transition-colors"
                    >
                      <X className="w-6 h-6 text-gray-600" />
                    </button>
                  </div>

                  <div className="p-6 overflow-y-auto">
                    <div className="space-y-4">
                      {analysisResult.isPlant ? (
                        // Tampilkan detail jika adalah tanaman
                        analysisResult.details && (
                          <>
                            <div>
                              <h4 className="text-gray-500 text-sm font-medium mb-2">INFORMASI ILMIAH</h4>
                              <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                                <p className="text-sm">
                                  <span className="font-medium">Nama Ilmiah:</span> {analysisResult.details.scientificName}
                                </p>
                                <p className="text-sm">
                                  <span className="font-medium">Familia:</span> {analysisResult.details.family}
                                </p>
                                <p className="text-sm">
                                  <span className="font-medium">Jenis:</span> {analysisResult.details.type}
                                </p>
                              </div>
                            </div>

                            <div>
                              <h4 className="text-gray-500 text-sm font-medium mb-2">KARAKTERISTIK</h4>
                              <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                                {analysisResult.details.characteristics && (
                                  <>
                                    <p className="text-sm">
                                      <span className="font-medium">Tinggi:</span> {analysisResult.details.characteristics.height}
                                    </p>
                                    <p className="text-sm">
                                      <span className="font-medium">Warna:</span> {analysisResult.details.characteristics.color}
                                    </p>
                                    <p className="text-sm">
                                      <span className="font-medium">Daun:</span> {analysisResult.details.characteristics.leafType}
                                    </p>
                                    {analysisResult.details.characteristics.flowerType && (
                                      <p className="text-sm">
                                        <span className="font-medium">Bunga:</span> {analysisResult.details.characteristics.flowerType}
                                      </p>
                                    )}
                                  </>
                                )}
                              </div>
                            </div>

                            {analysisResult.details.careInstructions && (
                              <div>
                                <h4 className="text-gray-500 text-sm font-medium mb-2">PANDUAN PERAWATAN</h4>
                                <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                                  <p className="text-sm">
                                    <span className="font-medium">Penyiraman:</span> {analysisResult.details.careInstructions.watering}
                                  </p>
                                  <p className="text-sm">
                                    <span className="font-medium">Sinar Matahari:</span> {analysisResult.details.careInstructions.sunlight}
                                  </p>
                                  <p className="text-sm">
                                    <span className="font-medium">Tanah:</span> {analysisResult.details.careInstructions.soil}
                                  </p>
                                  <p className="text-sm">
                                    <span className="font-medium">Suhu:</span> {analysisResult.details.careInstructions.temperature}
                                  </p>
                                </div>
                              </div>
                            )}
                          </>
                        )
                      ) : (
                        // Tampilkan pesan error jika bukan tanaman
                        <div className="bg-red-50 rounded-lg p-4">
                          <p className="text-red-600">{analysisResult.error}</p>
                          <p className="text-sm text-gray-600 mt-2">
                            Silakan coba ambil gambar lain dengan fokus pada tanaman yang ingin diidentifikasi.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="px-6 py-4 bg-gray-50 flex justify-end shrink-0">
                    <button
                      onClick={() => {
                        setShowModal(false)
                        setCapturedImage(null)
                        setAnalysisResult(null)
                      }}
                      className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800"
                    >
                      Tutup
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="h-[env(safe-area-inset-bottom)] bg-black" />
        </div>

        <Toaster />
      </div>

      <div className="hidden md:block fixed inset-0 bg-black -z-10" />
    </main>
  )
} 