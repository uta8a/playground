import scala.concurrent.{Future, Promise, ExecutionContext}
import scala.concurrent.duration.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory, TimeUnit}

final class AsyncBarrier(n0: Int)(using ec: ExecutionContext):
  require(n0 >= 0, s"n0 must be >= 0, but was $n0")

  /** 何人enter()したか */
  private val count = new AtomicInteger(0)
  /** 返すFutureの元になるもの */
  private val gate  = Promise[Unit]()

  // ---- public API ---------------------------------------------------------

  /** カウントを1つ増やす。n0に達した瞬間に待機中のFutureが完了する。 */
  def enter(): Future[Unit] =
    if n0 == 0 then Future.unit
    else
      val n = count.incrementAndGet()
      if n == n0 then gate.trySuccess(())
      gate.future

  /** 現在の残りカウントを返す（観測用）。 */
  def getCount: Int = count.get()
