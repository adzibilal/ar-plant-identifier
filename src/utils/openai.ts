import axios from 'axios'

const OPENAI_API_KEY = process.env.NEXT_PUBLIC_OPENAI_API_KEY

export const getPlantInfo = async (plantName: string) => {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'Anda adalah ahli botani yang memberikan informasi tentang tanaman.'
          },
          {
            role: 'user',
            content: `Berikan informasi singkat tentang tanaman ${plantName} dalam bahasa Indonesia.`
          }
        ]
      },
      {
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    )

    return response.data.choices[0].message.content
  } catch (error) {
    console.error('Error getting plant information:', error)
    return 'Maaf, tidak dapat mengambil informasi tanaman saat ini.'
  }
} 