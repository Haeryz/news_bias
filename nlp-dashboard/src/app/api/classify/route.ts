// File 1: /news_bias/nlp-dashboard/src/app/api/classify/route.ts
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { text } = await request.json();
    
    // Fixed: Added /classify endpoint to the URL
    const response = await fetch('https://brrbrrpatapim-dm.icystone-73750e36.southeastasia.azurecontainerapps.io/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error:', response.status, errorText);
      throw new Error(`Backend responded with ${response.status}: ${errorText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json({ 
      error: 'Failed to classify text',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}