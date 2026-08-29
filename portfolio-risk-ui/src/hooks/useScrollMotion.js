import { useEffect } from 'react'
import { initScrollMotion } from '../lib/motion/scroll-motion'

export function useScrollMotion(options) {
  useEffect(() => {
    const motion = initScrollMotion(options)
    return motion.destroy
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
