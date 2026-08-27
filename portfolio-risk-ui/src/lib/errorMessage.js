// FastAPI sends `detail` as a plain string for HTTPException, but as an array
// of Pydantic validation-error objects ({type, loc, msg, input}) for a 422 —
// normalise both into a renderable string so callers never get handed an object.
export function errorMessage(err, fallback) {
  const detail = err.response?.data?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail) && detail.length) {
    return detail.map(d => d?.msg || 'Invalid input').join('; ')
  }
  return fallback
}
