package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// TraceSlogHandler is a custom slog.Handler that adds trace context to log entries
type TraceSlogHandler struct {
	handler slog.Handler
}

// Ensure TraceSlogHandler implements slog.Handler
var _ slog.Handler = (*TraceSlogHandler)(nil)

// NewTraceSlogHandler creates a new slog handler that injects trace context
func NewTraceSlogHandler(handler slog.Handler) *TraceSlogHandler {
	return &TraceSlogHandler{handler: handler}
}

// Enabled implements slog.Handler.
func (h *TraceSlogHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.handler.Enabled(ctx, level)
}

// Handle implements slog.Handler.
func (h *TraceSlogHandler) Handle(ctx context.Context, record slog.Record) error {
	// Get the current span context from the context
	spanContext := trace.SpanContextFromContext(ctx)
	if spanContext.IsValid() {
		// Add trace ID and span ID to the log record if there's a valid span
		record.AddAttrs(
			slog.String("trace_id", spanContext.TraceID().String()),
			slog.String("span_id", spanContext.SpanID().String()),
		)
	}
	return h.handler.Handle(ctx, record)
}

// WithAttrs implements slog.Handler.
func (h *TraceSlogHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return NewTraceSlogHandler(h.handler.WithAttrs(attrs))
}

// WithGroup implements slog.Handler.
func (h *TraceSlogHandler) WithGroup(name string) slog.Handler {
	return NewTraceSlogHandler(h.handler.WithGroup(name))
}

// initTracer sets up the OpenTelemetry tracer
func initTracer() (*sdktrace.TracerProvider, error) {
	// Create stdout exporter to send traces to stdout
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, err
	}

	// Create a new trace provider with the exporter
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String("slog-otel-example"),
			semconv.ServiceVersionKey.String("0.1.0"),
		)),
	)

	// Set the global trace provider
	otel.SetTracerProvider(tp)
	return tp, nil
}

func main() {
	// Initialize the tracer provider
	tp, err := initTracer()
	if err != nil {
		slog.Error("Failed to initialize tracer", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			slog.Error("Error shutting down tracer provider", "error", err)
		}
	}()

	// Get a tracer from the provider
	tracer := tp.Tracer("slog-otel-example")

	// Create a basic JSON handler
	jsonHandler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})

	// Wrap with our trace handler
	traceHandler := NewTraceSlogHandler(jsonHandler)

	// Set the default logger with our trace handler
	slog.SetDefault(slog.New(traceHandler))

	// Create a parent span
	ctx, span := tracer.Start(context.Background(), "main")
	defer span.End()

	// Add some attributes to the span
	span.SetAttributes(attribute.String("app", "slog-otel-example"))

	// Log within the span context
	slog.InfoContext(ctx, "Application started", "time", time.Now())

	// Create a child span for the processWork function
	processWork(ctx, tracer)

	slog.InfoContext(ctx, "Application finished", "time", time.Now())
}

func processWork(ctx context.Context, tracer trace.Tracer) {
	// Create a child span
	ctx, span := tracer.Start(ctx, "processWork")
	defer span.End()

	// Add some attributes
	span.SetAttributes(attribute.Int("items_processed", 42))

	// Log with the current span context
	slog.InfoContext(ctx, "Processing work started")

	// Simulate some work
	time.Sleep(100 * time.Millisecond)

	// Create another child span for a sub-operation
	doSubOperation(ctx, tracer)

	slog.InfoContext(ctx, "Processing work completed", "status", "success")
}

func doSubOperation(ctx context.Context, tracer trace.Tracer) {
	// Create a nested span
	ctx, span := tracer.Start(ctx, "doSubOperation")
	defer span.End()

	// Simulate an error
	span.SetAttributes(attribute.Bool("error", true))

	// Log with the error
	slog.ErrorContext(ctx, "Error in sub-operation", 
		"error", "something went wrong",
		"operation_id", "12345")

	// Simulate more work
	time.Sleep(50 * time.Millisecond)
}
