import { useState } from 'react'

export function useToast() {
    const toast = ({ title, description, variant }: { title: string, description?: string, variant?: string }) => {
        console.log(`TOAST: ${title} - ${description} [${variant}]`)
        // Minimal mock for now, can be expanded to a context-based system if needed
    }
    return { toast }
}
