package utils

func EnsurePositiveInt(value int64, defaultValue int64) int64 {
	if value <= 0 {
		return defaultValue
	}
	return value
}

func EnsurePositiveFloat(value float64, defaultValue float64) float64 {
	if value <= 0 {
		return defaultValue
	}
	return value
}
