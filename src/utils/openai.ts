import OpenAI from 'openai'
import { ChatCompletionContentPart } from 'openai/resources/chat/completions'

const OPENAI_API_KEY = process.env.NEXT_PUBLIC_OPENAI_API_KEY

// Interface untuk format JSON yang diharapkan dari OpenAI
interface PlantAnalysisResponse {
  isPlant: boolean
  name?: string
  description?: string
  confidence: number
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
  error?: string
}

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  dangerouslyAllowBrowser: true // Karena kita menggunakan di browser
})

export const analyzePlantImage = async (base64Image: string): Promise<PlantAnalysisResponse> => {
  if (!OPENAI_API_KEY) {
    throw new Error('OpenAI API key tidak ditemukan')
  }

  try {
    // Pastikan base64Image memiliki format yang benar
    const formattedBase64 = base64Image.startsWith('data:image') 
      ? base64Image 
      : `data:image/jpeg;base64,${base64Image}`

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'Anda adalah sistem analisis gambar yang fokus pada identifikasi tanaman. Berikan respons dalam format JSON yang valid. Jangan sertakan backtick, komentar, atau karakter lain selain JSON murni. Pastikan semua respons dalam Bahasa Indonesia, confidence dalam bentuk angka 0-100, dan berikan informasi selengkap mungkin jika terdeteksi tanaman.'
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analisis gambar ini dan berikan respons dalam format berikut jika terdeteksi tanaman: {"isPlant":true,"name":"Nama umum tanaman","description":"Deskripsi singkat","confidence":95,"details":{"scientificName":"Nama ilmiah","family":"Familia","type":"Jenis tanaman","habitat":"Habitat alami","characteristics":{"height":"Tinggi","color":"Warna","leafType":"Jenis daun","flowerType":"Jenis bunga"},"uses":["Kegunaan 1","Kegunaan 2"],"careInstructions":{"watering":"Panduan penyiraman","sunlight":"Kebutuhan sinar","soil":"Jenis tanah","temperature":"Suhu"}}}. Jika bukan tanaman: {"isPlant":false,"confidence":90,"error":"Tidak dapat mengidentifikasi tanaman"}'
            } as ChatCompletionContentPart,
            {
              type: 'image_url',
              image_url: {
                url: formattedBase64,
                detail: 'high'
              }
            } as ChatCompletionContentPart
          ]
        }
      ],
      max_tokens: 1500,
      temperature: 0.5,
      stream: false,
      n: 1
    })

    const content = response.choices[0].message.content
    if (!content) {
      throw new Error('Tidak ada respons dari OpenAI')
    }

    try {
      // Tunggu sebentar untuk memastikan response lengkap
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const cleanContent = content
        .trim()
        .replace(/\n/g, ' ') // Hapus newline
        .replace(/\s+/g, ' ') // Hapus multiple spaces
      
      return JSON.parse(cleanContent)
    } catch (parseError) {
      console.error('Error parsing JSON:', parseError)
      console.debug('Raw content:', content)
      return {
        isPlant: false,
        confidence: 0,
        error: 'Gagal memproses respons dari AI'
      }
    }

  } catch (error) {
    console.error('Error analyzing plant image:', error)
    return {
      isPlant: false,
      confidence: 0,
      error: 'Terjadi kesalahan saat menganalisis gambar'
    }
  }
} 