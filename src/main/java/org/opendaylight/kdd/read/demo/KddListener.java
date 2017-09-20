/*
 * Copyright (c) 2015 Tata Consultancy Services, Inc. and others.  All rights reserved.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v1.0 which accompanies this distribution,
 * and is available at http://www.eclipse.org/legal/epl-v10.html
 */
package org.opendaylight.kdd.read.demo;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 *
 * @author rashmi.tomer@tcs.com
 *
 */
public class KddListener implements IterationListener {

	private static final long serialVersionUID = 1L;
	private int printIterations = 10;
	private boolean invoked = false;

	public KddListener(int printIterations) {
		this.printIterations = printIterations;
	}

	public KddListener() {
	}

	@Override
	public boolean invoked() {
		return invoked;
	}

	@Override
	public void invoke() {
		this.invoked = true;
	}

	@Override
	public void iterationDone(Model model, int iteration) {
		if (printIterations <= 0)
			printIterations = 1;
		if (iteration % printIterations == 0) {
			invoke();
			double result = model.score();
			
			model.applyLearningRateScoreDecay();
			//System.out.println("Value of Loss funtion at iteration " + iteration + " is " + result);
			

		}

	}

}
