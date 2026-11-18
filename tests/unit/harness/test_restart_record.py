from app.harness.run_controller import RunController

def test_H16_abandoned_write_replay_after_restart_is_unknown_never_redispatched(env):
    first=RunController(env['b'],env['cfg'])
    ctx,_=first.begin(env['a'],env['conv'],'conv:'+env['conv'],{'message':'handoff'},'restart-key')
    first.update(ctx,status='RUNNING',write_started=1)
    restarted=RunController(env['b'],env['cfg'])
    context,response=restarted.begin(env['a'],env['conv'],'conv:'+env['conv'],{'message':'handoff'},'restart-key')
    assert context is None and response['invocation_status']=='UNKNOWN_OUTCOME'
    assert response['invocation_id']==ctx.invocation_id
    assert not restarted.tasks
