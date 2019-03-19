import edu.cmu.dynet._
import org.clulab.embeddings.word2vec.Word2Vec

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Source
import scala.util.Random
import scala.collection.mutable

import org.clulab.fatdynet.utils.CloseableModelSaver
import org.clulab.fatdynet.utils.Closer.AutoCloser
import org.clulab.utils.{MathUtils, Serializer}

import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.io.{FileWriter, PrintWriter}


object Hypernym {
  val HIDDEN_SIZE = 100
  val ITERATIONS = 30
  val EMBEDDING_SIZE = 200

  // This is a little fancy because it works with both String and Char keys.
  class ByLineMapBuilder[KeyType](val converter: String => KeyType) extends ByLineBuilder {
    val mutableMap: mutable.Map[KeyType, Int] = new mutable.HashMap

    def addLine(line: String): Unit = {
      val Array(key, value) = line.split('\t')

      mutableMap += ((converter(key), value.toInt))
    }

    def toValue: Map[KeyType, Int] = mutableMap.toMap
  }

  // This only works with Strings.
  class ByLineArrayBuilder extends ByLineBuilder {
    val arrayBuffer: ArrayBuffer[String] = ArrayBuffer.empty

    def addLine(line: String): Unit = {
      arrayBuffer += line
    }

    def toValue: Array[String] = arrayBuffer.toArray
  }

  // This only works with Strings.
  class ByLineIntBuilder extends ByLineBuilder {
    var value: Option[Int] = None

    def addLine(line: String): Unit = {
      value = Some(line.toInt)
    }

    def toValue: Int = value.get
  }

  trait ByLineBuilder {
    def addLine(line: String): Unit
  }

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

  protected def save[T](printWriter: PrintWriter, values: Map[T, Int], comment: String): Unit = {
    printWriter.println("# " + comment)
    values.foreach { case (key, value) =>
      printWriter.println(s"$key\t$value")
    }
    printWriter.println() // Separator
  }

  protected def save[T](printWriter: PrintWriter, values: Array[T], comment: String): Unit = {
    printWriter.println("# " + comment)
    values.foreach(printWriter.println)
    printWriter.println() // Separator
  }

  protected def save[T](printWriter: PrintWriter, value: Long, comment: String): Unit = {
    printWriter.println("# " + comment)
    printWriter.println(value)
    printWriter.println() // Separator
  }
  def save(modelFilename: String, w2i:Map[String, Int], m: ParameterCollection, lookupParameters:LookupParameter):Unit = {
    val dynetFilename = modelFilename + ".rnn"
    val x2iFilename = modelFilename + ".x2i"

    new CloseableModelSaver(dynetFilename).autoClose { modelSaver =>
      modelSaver.addModel(m, "/all")
    }

    Serializer.using(new PrintWriter(new OutputStreamWriter(new BufferedOutputStream(new FileOutputStream(x2iFilename)), "UTF-8"))) { printWriter =>
      save(printWriter, w2i, "w2i")
      val dim = lookupParameters.dim().get(0)
      save(printWriter, dim, "dim")
    }
  }

  def load(filename: String, byLineBuilders: Array[ByLineBuilder]): Unit = {
    var expectingComment = true
    var byLineBuilderIndex = 0

    Serializer.using(Source.fromFile(filename, "UTF-8")) { source =>
      source.getLines.foreach { line =>
        if (line.nonEmpty)
          if (expectingComment)
            expectingComment = false
          else
            byLineBuilders(byLineBuilderIndex).addLine(line)
        else {
          byLineBuilderIndex += 1
          expectingComment = true
        }
      }
    }
  }

  def main(args: Array[String]) {
    Initialize.initialize()
    println("Dynet initialized!")
    val m = new ParameterCollection
    val sgd = new SimpleSGDTrainer(m)
    ComputationGraph.renew()

    val words = new ListBuffer[String]
    val filename = "1A.english.vocabulary.txt"
    for (line <- Source.fromFile(filename).getLines) {
      words += line.stripLineEnd.toLowerCase()
    }
    val w2i = words.sorted.zipWithIndex.toMap
    val embeddingsFile = "glove.6B.200d.txt"
    val lookupParameters = m.addLookupParameters(400000, Dim(200, 1))
    val w2v = new Word2Vec(embeddingsFile) // Some(w2i.keySet))
    val unknownEmbed = new Array[Double](EMBEDDING_SIZE)
    for(i <- unknownEmbed.indices) unknownEmbed(i) = 0.0

    var unknownCount = 0
    for(word <-  w2i.keySet) {
      if (word.contains(" ")) {
        val vec = new Array[Double](EMBEDDING_SIZE)
        for (w<-word.split(" ")){
          if (w2v.matrix.contains(w)){
            for (i<-vec.indices){
              vec(i) += w2v.matrix(w)(i)
            }
          }
        }
        for (i<-vec.indices){
          vec(i) /= word.split(" ").length
        }
        lookupParameters.initialize(w2i(word), new FloatVector(toFloatArray(vec)))
      } else {
        if (w2v.matrix.contains(word)) {
          lookupParameters.initialize(w2i(word), new FloatVector(toFloatArray(w2v.matrix(word))))
        } else {
          unknownCount += 1
        }
      }
    }


    var p_Phi = Seq[Parameter]()
    for (_ <- 1 to 24){
      p_Phi = p_Phi :+ m.addParameters(Dim(200, 200))
    }
    val p_W = m.addParameters(Dim(1, 24))
    val p_b = m.addParameters(Dim(1))

    var Phi = Seq[Expression]()
    for (i <- 0 to 23){
      Phi = Phi :+ Expression.parameter(p_Phi(i))
    }
    val W = Expression.parameter(p_W)
    val b = Expression.parameter(p_b)

    val e_q_values = new IntPointer
    e_q_values.set(0)
    val e_q = Expression.lookup(lookupParameters, e_q_values.value())
    val e_h_values = new IntPointer
    e_h_values.set(0)
    val e_h = Expression.lookup(lookupParameters, e_h_values.value())
    val t_value = new FloatPointer
    t_value.set(0)
    val t = Expression.input(t_value)


    var P_seq = Seq[Expression]()
    for (i <- 0 to 23){
      P_seq = P_seq :+ Phi(i)*e_q
    }
    val P = Expression.transpose(Expression.concatenateCols(ExpressionVector.Seq2ExpressionVector(P_seq)))
    val s = P * e_h
    val y = Expression.logistic(W * s + b)

    val loss_expr = - t * Expression.log(y) - (1-t) * Expression.log(1-y)

    println()
    println("Computation graphviz structure:")
    ComputationGraph.printGraphViz()
    println()
    println("Training...")
    var trainning_data = new ListBuffer[datapoint]()
    for(line <- Source.fromFile(args(0)).getLines) {
      val datapoints = line.split("\t")
      for (d<-datapoints){
        var temp = d.replace("(","")
        temp = temp.replace(")","")
        val datapoint = new datapoint()
        datapoint.hypo = temp.split(",")(0).toLowerCase().stripLineEnd
        datapoint.hyper = temp.split(",")(1).toLowerCase().stripLineEnd
        datapoint.is_pos = true
        trainning_data += datapoint
      }
    }
    for(line <- Source.fromFile(args(1)).getLines) {
      val datapoints = line.split("\t")
      for (d<-datapoints){
        var temp = d.replace("(","")
        temp = temp.replace(")","")
        val datapoint = new datapoint()
        datapoint.hypo = temp.split(",")(0).toLowerCase().stripLineEnd
        datapoint.hyper = temp.split(",")(1).toLowerCase().stripLineEnd
        datapoint.is_pos = false
        trainning_data += datapoint
      }
    }
    val random = new Random()
    trainning_data = random.shuffle(trainning_data)
    for (iter <- 0 to ITERATIONS - 1) {
      var loss: Float = 0
      for (d <- trainning_data) {
        val q = d.hypo
        val h = d.hyper
        val l = d.is_pos
        e_q_values.set(w2i(q))
        e_h_values.set(w2i(h))
        t_value.set(if (l) 1 else 0)
        loss += ComputationGraph.forward(loss_expr).toFloat
        ComputationGraph.backward(loss_expr)
        sgd.update()
      }
      save(s"model$iter", w2i, m, lookupParameters)
      sgd.learningRate *= 0.998f
      println("iter = " + iter + ", loss = " + loss)
    }
  }
}