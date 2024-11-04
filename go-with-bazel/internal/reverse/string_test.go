package reverse_test

import (
	"os"
	"testing"

	"github.com/uta8a/playground/go-with-bazel/internal/reverse"
)

func TestString(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
	}{
		{
			name: "empty",
		},
		{
			name: "single",
		},
		{
			name: "multiple",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup
			givenBytes, err := os.ReadFile("testdata/" + tt.name + "_string_input.txt")
			if err != nil {
				t.Fatal(err)
			}
			expectedBytes, err := os.ReadFile("testdata/" + tt.name + "_string_output.txt")
			if err != nil {
				t.Fatal(err)
			}
			given := string(givenBytes)
			expected := string(expectedBytes)

			// Exercise
			actual := reverse.String(string(given))

			// Verify
			if actual != expected {
				t.Errorf("expected: %s, actual: %s", expected, actual)
			}
		})
	}
}
