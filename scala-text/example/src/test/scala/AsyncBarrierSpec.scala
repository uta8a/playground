// src/test/scala/AsyncCountDownLatchFlatSpec.scala
import org.scalatest.flatspec.AsyncFlatSpec
import scala.concurrent.{Future, ExecutionContext, Promise}
import scala.concurrent.duration.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory, TimeUnit}

class AsyncBarrierSpec extends AsyncFlatSpec: 

  given ExecutionContext = ExecutionContext.global

  "await" should "complete only after all enter calls" in {
    val b = new AsyncBarrier(3)
    Future { Thread.sleep(50);  b.enter() }
    Future { Thread.sleep(80);  b.enter() }
    Future { Thread.sleep(120); b.enter() }
    b.enter().map { _ =>
      succeed
    }
  }

