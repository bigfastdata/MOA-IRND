/*
 *    AbsurdOracleClassifier.java
 *    Copyright (C) 2013 Brandon S. Parker
 *    @author Brandon S. Parker (brandon.parker@utdallas.edu)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 *    
 */
package moa.classifiers.functions;

import java.util.Random;
import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.options.FloatOption;
import moa.options.IntOption;
import weka.core.Instance;

/**
 * As an 'oracle' perfect black-box classifier, this method lets you
 * test with a given target accuracy (useful for meta-learner testing).
 * 
 * @author Brandon Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class AbsurdOracleClassifier extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    public FloatOption desiredAccuracyOption = new FloatOption("percentCorrect",
            'p', "Desired Accuracy",
            1.0, 0.00, 1.00);
    
    public IntOption myLocalRandomSeedOption = new IntOption("rngSeed",
            's', "RNG Seed",
            42, 1, Integer.MAX_VALUE);
    
    private Random randNumGen;
    
    @Override
    public String getPurposeString() {
        return "Always predicts the right class (but cheats).";
    }

    @Override
    public void resetLearningImpl() {
        this.randNumGen = new Random();
        this.randNumGen.setSeed(this.myLocalRandomSeedOption.getValue());
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
       
    }

    @Override
    public double[] getVotesForInstance(Instance i) {
        DoubleVector observedClassDistribution = new DoubleVector();
        if (this.randNumGen.nextFloat() < this.desiredAccuracyOption.getValue())
        {
            observedClassDistribution.addToValue((int) i.classValue(), i.weight());
        }
        else
        {
            observedClassDistribution.addToValue(((int) i.classValue() + 1)% i.numClasses(), i.weight());
            
        }
        return observedClassDistribution.getArrayCopy();
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "Oracle Predicted (always right - or at least at the provided accuracy parameter). Intended to help test meta learners.");
    }

    public boolean isRandomizable() {
        return false;
    }
}
