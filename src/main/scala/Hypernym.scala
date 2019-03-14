import edu.cmu.dynet._
import org.clulab.embeddings.word2vec.Word2Vec
import scala.collection.mutable.ListBuffer
import scala.io.Source


object Hypernym {
  val HIDDEN_SIZE = 100
  val ITERATIONS = 30
  val EMBEDDING_SIZE = 200

  protected def add(dst:Array[Double], src:Array[Double]): Unit = {
    assert(dst.length == src.length)
    for(i <- dst.indices) {
      dst(i) += src(i)
    }
  }

  protected def toFloatArray(doubles: Array[Double]): Array[Float] = {
    val floats = new Array[Float](doubles.length)
    for (i <- doubles.indices) {
      floats(i) = doubles(i).toFloat
    }
    floats
  }

  def main(args: Array[String]) {
    val embeddingsFile = args(0)
    println("Running XOR example")
    Initialize.initialize()
    println("Dynet initialized!")
    val m = new ParameterCollection
    val sgd = new SimpleSGDTrainer(m)
    ComputationGraph.renew()

    val words = new ListBuffer[String]
    val filename = args(1)
    for (line <- Source.fromFile(filename).getLines) {
      words += line.stripLineEnd.toLowerCase()
    }
    val w2i = words.sorted.zipWithIndex.toMap

    val lookupParameters = m.addLookupParameters(400000, Dim(200, 1))
    val w2v = new Word2Vec(embeddingsFile) // Some(w2i.keySet))
    val unknownEmbed = new Array[Double](EMBEDDING_SIZE)
    for(i <- unknownEmbed.indices) unknownEmbed(i) = 0.0

    var unknownCount = 0
    for(word <- w2v.matrix.keySet){// w2i.keySet) {
      if(w2i.contains(word)) {
        lookupParameters.initialize(w2i(word), new FloatVector(toFloatArray(w2v.matrix(word))))
      } else {
        add(unknownEmbed, w2v.matrix(word))
        unknownCount += 1
      }
    }
    for(i <- unknownEmbed.indices) {
      unknownEmbed(i) /= unknownCount
    }

    val p_Phi = Seq[Parameter]()
    for (_ <- 1 to 24){
      p_Phi :+ m.addParameters(Dim(HIDDEN_SIZE, 200))
    }
    val p_W = m.addParameters(Dim(1, HIDDEN_SIZE))
    val p_b = m.addParameters(Dim(1))

    val Phi = Seq[Expression]()
    for (i <- 1 to 24){
      Phi :+ Expression.parameter(p_Phi(i))
    }
    val W = Expression.parameter(p_W)
    val b = Expression.parameter(p_b)

    val e_q_values = new FloatVector(200)
    val e_q = Expression.input(Dim(200), e_q_values)
    val e_h_values = new FloatVector(200)
    val e_h = Expression.input(Dim(200), e_h_values)
    val t_value = new FloatPointer
    t_value.set(0)
    val t = Expression.input(t_value)


    val P_seq = Seq[Expression]()
    for (i <- 1 to 24){
      P_seq :+ Expression.transpose(Phi(i)*e_q)
    }
    val P = Expression.concatenateCols(ExpressionVector.Seq2ExpressionVector(P_seq))
    val s = P * e_h
    val y = Expression.logistic(W * s + b)

    val loss_expr = t * Expression.log(y) + (1-t) * Expression.log(1-y)

    println()
    println("Computation graphviz structure:")
    ComputationGraph.printGraphViz()
    println()
    println("Training...")

    for (iter <- 0 to ITERATIONS - 1) {
      var loss: Float = 0
      for (mi <- 0 to 3) {
        val x1: Boolean = mi % 2 > 0
        val x2: Boolean = (mi / 2) % 2 > 0
        e_q_values.update(0, if (x1) 1 else -1)
        e_h_values.update(1, if (x2) 1 else -1)
        t_value.set(if (x1 != x2) 1 else -1)
        loss += ComputationGraph.forward(loss_expr).toFloat
        ComputationGraph.backward(loss_expr)
        sgd.update()
      }
      sgd.learningRate *= 0.998f
      loss /= 4
      println("iter = " + iter + ", loss = " + loss)
    }
  }
}
