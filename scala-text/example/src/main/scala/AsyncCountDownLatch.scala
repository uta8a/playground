import scala.concurrent.{Future, Promise, ExecutionContext}
import scala.concurrent.duration.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory, TimeUnit}

final class AsyncCountDownLatch(initial: Int)(using ec: ExecutionContext):
  require(initial >= 0, s"initial must be >= 0, but was $initial")

  private val remaining = new AtomicInteger(initial)
  private val gate      = Promise[Unit]()

  // ---- public API ---------------------------------------------------------

  /** カウントを1つ減らす。0に達した瞬間に待機中のFutureが完了する。 */
  def countDown(): Unit =
    val after = remaining.updateAndGet(n => if n > 0 then n - 1 else 0)
    if after == 0 then gate.trySuccess(())

  /** 現在の残りカウントを返す（観測用）。 */
  def getCount: Int = remaining.get()

  /** すべてカウントが終わると成功になる Future。 */
  def await(): Future[Unit] =
    if remaining.get() == 0 then Future.unit else gate.future

  /**
   * タイムアウト付き待機。期限までに 0 になれば true、間に合わなければ false。
   * スレッドをブロックしない（内部でスケジューラが Promise を攻める）。
   */
  def tryAwait(timeout: FiniteDuration): Future[Boolean] =
    if remaining.get() == 0 then Future.successful(true)
    else
      val done    = gate.future.map(_ => true)
      val timedOut = Scheduler.after(timeout).map(_ => false)
      Future.firstCompletedOf(Seq(done, timedOut))

// ---- 共有スケジューラ（デーモン1本） --------------------------------------
private object Scheduler:
  private val tf: ThreadFactory = (r: Runnable) =>
    val t = new Thread(r, "async-latch-scheduler")
    t.setDaemon(true); t
  private val es = Executors.newSingleThreadScheduledExecutor(tf)

  def after(d: FiniteDuration): Future[Unit] =
    val p = Promise[Unit]()
    es.schedule(new Runnable { def run(): Unit = p.trySuccess(()) }, d.toMillis, TimeUnit.MILLISECONDS)
    p.future

