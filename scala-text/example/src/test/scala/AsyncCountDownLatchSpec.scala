// src/test/scala/AsyncCountDownLatchFlatSpec.scala
import org.scalatest.flatspec.AsyncFlatSpec
import scala.concurrent.{Future, ExecutionContext, Promise}
import scala.concurrent.duration.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory, TimeUnit}

class AsyncCountDownLatchFlatSpec extends AsyncFlatSpec: 

  given ExecutionContext = ExecutionContext.global

  "await" should "complete only after all countDown calls" in {
    val latch = new AsyncCountDownLatch(3)
    Future { Thread.sleep(50);  latch.countDown() }
    Future { Thread.sleep(80);  latch.countDown() }
    Future { Thread.sleep(120); latch.countDown() }
    latch.tryAwait(200.millis).map { _ =>
      assert(latch.getCount === 0)
    }
  }

