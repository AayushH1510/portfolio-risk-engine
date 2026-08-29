import { useEffect } from 'react'
import { createSmokeField } from '../lib/motion/smoke-field'

export function useSmokeField(canvasRef, options) {
  useEffect(() => {
    if (!canvasRef.current) return
    const field = createSmokeField(canvasRef.current, options)
    return field.destroy
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
