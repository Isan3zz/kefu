"""Adapter only: the existing Recorder remains the sole durable trace writer."""
from app.observability import active_trace,clean

class Telemetry:
    def __init__(self,ctx,enabled=True): self.ctx,self.enabled=ctx,enabled
    def emit(self,kind,**attributes):
        recorder=active_trace.get()
        if recorder is None or not self.enabled: return None
        safe=clean(attributes)
        fields=dict(schema_version='1.1',trace_id=self.ctx.trace_id,invocation_id=self.ctx.invocation_id,evaluation_run_id=self.ctx.evaluation_run_id,evaluation_case_id=self.ctx.evaluation_case_id,conversation_id=self.ctx.conversation_id,graph_thread_id=self.ctx.graph_thread_id)
        fields.update(safe)
        return recorder.emit(kind,attributes=safe,**fields)
