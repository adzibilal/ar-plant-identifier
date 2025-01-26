'use client'

import { useEffect, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as cocossd from '@tensorflow-models/coco-ssd'
import { getPlantInfo } from '@/utils/openai'
import { Camera } from 'lucide-react'

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

      const predictions = await detectorModel.detect(video)
      
      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.height)
        
        predictions
          .filter(pred => ['potted plant', 'plant', 'tree', 'flower'].includes(pred.class))
          .forEach(prediction => {
            const [x, y, width, height] = prediction.bbox

            context.strokeStyle = '#00ff00'
            context.lineWidth = 2
            context.strokeRect(x, y, width, height)

            context.fillStyle = '#00ff00'
            context.fillRect(x, y - 20, prediction.class.length * 8, 20)
            context.fillStyle = '#000000'
            context.font = '16px Arial'
            context.fillText(prediction.class, x, y - 5)
          })
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

  const captureAndPredict = async () => {
    if (!model || !webcamRef.current || isAnalyzing) return

    try {
      setIsAnalyzing(true)
      const imageElement = webcamRef.current.getScreenshot()
      
      if (imageElement) {
        const img = new Image()
        img.src = imageElement
        img.onload = async () => {
          try {
            const predictions = await model.classify(img)
            if (predictions && predictions[0]) {
              const newPrediction = predictions[0].className
              setPrediction(newPrediction)
              
              const info = await getPlantInfo(newPrediction)
              setPlantInfo(info)
            }
          } catch (error) {
            console.error('Error predicting:', error)
          } finally {
            setIsAnalyzing(false)
          }
        }
      }
    } catch (error) {
      console.error('Error capturing image:', error)
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
    <main className="fixed inset-0 bg-black">
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

        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10">
          <button
            onClick={captureAndPredict}
            disabled={isAnalyzing}
            className={`w-16 h-16 rounded-full flex items-center justify-center
              ${isAnalyzing 
                ? 'bg-gray-500' 
                : 'bg-white hover:bg-gray-200 active:bg-gray-300'
              } transition-colors`}
          >
            {isAnalyzing ? (
              <div className="w-8 h-8 border-4 border-gray-800 border-t-transparent rounded-full animate-spin" />
            ) : (
              <Camera className="w-8 h-8 text-black cursor-pointer" />
            )}
          </button>
        </div>

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6 pb-28">
          <div className="max-w-3xl mx-auto">
            {prediction && (
              <p className="text-white text-xl font-bold mb-2">
                Terdeteksi: {prediction}
              </p>
            )}
            {plantInfo && (
              <div className="text-white/90 text-sm leading-relaxed">
                <p>{plantInfo}</p>
              </div>
            )}
          </div>
        </div>

        <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/80 to-transparent p-6">
          <h1 className="text-white text-xl font-bold text-center">
            AR Plant Identifier
          </h1>
        </div>
      </div>
    </main>
  )
} 