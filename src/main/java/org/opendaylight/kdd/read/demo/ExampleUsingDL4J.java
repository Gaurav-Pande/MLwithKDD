/*
 * Copyright (c) 2015 Tata Consultancy Services, Inc. and others.  All rights reserved.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v1.0 which accompanies this distribution,
 * and is available at http://www.eclipse.org/legal/epl-v10.html
 */
package org.opendaylight.kdd.read.demo;

import java.io.File;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author rashmi.tomer@tcs.com
 *
 */
public class ExampleUsingDL4J {

	private static Logger log = LoggerFactory.getLogger(ExampleUsingDL4J.class);

	private static final String source = "src/main/resources/classification/";
	private static final String csvFileNameEvalData = "data_eval_1.csv";
	private static final String csvFileNameTrainData = "data_train_1.csv";

	private static final String csvFilePathTrainData = source
			+ csvFileNameTrainData;
	private static final String csvFilePathEvalData = source
			+ csvFileNameEvalData;

	public static void main(String[] args) throws Exception {
		int seed = 6;
		double learningRate = 0.005;
		int batchSize = 50;
		int nEpochs = 50;

		int numInputs = 17;
		int numOutputs = 3;
		int numHiddenNodes = 20;

		// Load the training data:
		RecordReader rr = new CSVRecordReader();

		rr.initialize(new FileSplit(new File(csvFilePathTrainData)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,
				batchSize, 0, 3);

		// Load the test/evaluation data:
		RecordReader rrTest = new CSVRecordReader();

		rrTest.initialize(new FileSplit(new File(csvFilePathEvalData)));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,
				batchSize, 0, 3);

		// log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(1)
				.optimizationAlgo(
						OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRate)
				.updater(Updater.NESTEROVS)
				.momentum(0.9)
				.regularization(true)
				.l2(1e-4)
				.list(3)
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs)
								.nOut(numHiddenNodes)
								.weightInit(WeightInit.XAVIER)
								.activation("relu").build())
				.layer(1,
						new DenseLayer.Builder().nIn(numHiddenNodes)
								.nOut(numHiddenNodes)
								.weightInit(WeightInit.XAVIER)
								.activation("relu").build())
				.layer(2,
						new OutputLayer.Builder(
								LossFunction.NEGATIVELOGLIKELIHOOD)
								.weightInit(WeightInit.XAVIER)
								.activation("softmax").nIn(numHiddenNodes)
								.nOut(numOutputs).build()).pretrain(false)
				.backprop(true).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}

		System.out.println("Evaluate model....");
		Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatureMatrix();
			INDArray labels = t.getLabels();
			INDArray predicted = model.output(features, false);
			eval.eval(labels, predicted);
		}

		// Print the evaluation statistics
		System.out.println(eval.stats());

		// ------------------------------------------------------------------------------------
		// Training is complete. Code that follows is for plotting the data &
		// predictions only

		// Plot the data
		double xMin = -1.5;
		double xMax = 2.5;
		double yMin = -1;
		double yMax = 1.5;

		// Let's evaluate the predictions at every point in the x/y input space,
		// and plot this in the background
		int nPointsPerAxis = 100;
		double[][] evalPoints = new double[nPointsPerAxis * nPointsPerAxis][17];
		int count = 0;
		for (int i = 0; i < nPointsPerAxis - 1; i++) {
			for (int j = 0; j < nPointsPerAxis; j++) {
				double x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin;
				double y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin;

				evalPoints[count][0] = x;
				evalPoints[count][1] = y;

				count++;
			}
		}

		INDArray allXYPoints = Nd4j.create(evalPoints);
		INDArray predictionsAtXYPoints = model.output(allXYPoints);

		// Get all of the training data in a single array, and plot it:
		rr.initialize(new FileSplit(new File(csvFilePathTrainData)));
		rr.reset();
		int nTrainPoints = 21;// 2000;
		trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 3);
		DataSet ds = trainIter.next();
		PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(),
				allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		// Get test data, run the test data through the network to generate
		// predictions, and plot those predictions:

		rrTest.initialize(new FileSplit(new File(csvFilePathEvalData)));
		rrTest.reset();
		int nTestPoints = 11;// 1000;
		testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 3);
		ds = testIter.next();
		INDArray testPredicted = model.output(ds.getFeatures());
		PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted,
				allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		System.out
				.println("****************Example finished********************");
	}
}
