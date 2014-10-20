/*
 * Copyright 2014 bparker.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package moa.classifiers.novelClass;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import moa.classifiers.AbstractClassifier;
import static moa.classifiers.AbstractClassifier.contextIsCompatible;
import moa.core.InstancesHeader;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Provides additional checks and capabilities for novel class detection
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public abstract class AbstractNovelClassClassifier extends AbstractClassifier {
    private static final long serialVersionUID = 1L;
    protected Map<Integer,Integer> labelsSeen = new HashMap<>();
    protected int novelLabelIndex = 0;
    public static final String NOVEL_LABEL_STR = "_NOVEL_LABEL_";
    public static final String OUTLIER_LABEL_STR = "_OUTLIER_LABEL_";
    public static final String NOVEL_CLASS_INSTANCE_RELATIONSHIP_TYPE = "NCDIRT";
    
    
    @Override
    public String getPurposeString() {
        return "MOA Novel Class Detection Classifier: " + getClass().getCanonicalName();
    }
    
    @Override
    public void resetLearning() {
        this.trainingWeightSeenByModel = 0.0;
        if (isRandomizable()) {
            this.classifierRandom = new Random(this.randomSeed);
        }
        resetLearningImpl();
        this.labelsSeen.clear();
        this.novelLabelIndex = 0;
    }
    
    @Override
    public void setModelContext(InstancesHeader ih) {
        if (ih == null) {
            throw new IllegalArgumentException(
                    "Null model provided");
        }
        if (ih.classIndex() < 0) {
            throw new IllegalArgumentException(
                    "Context for a classifier must include a class to learn");
        }
        if (trainingHasStarted()
                && (this.modelContext != null)
                && (!contextIsCompatible(this.modelContext, ih))) {
            throw new IllegalArgumentException(
                    "New context is not compatible with existing model");
        }
        this.modelContext = ih;
        this.novelLabelIndex = ih.numClasses();
    }
    
    @Override
    public boolean correctlyClassifies(Instance inst) {
        double[] votes = getVotesForInstance(inst);
        int prediction = Utils.maxIndex(votes);
        int groundTruth = (int) inst.classValue();
        boolean isCorrect =  (prediction == groundTruth);
        if (!isCorrect && votes.length > inst.numClasses()) {   // If it was intially wrong, 
               if(!this.labelsSeen.containsKey(groundTruth)     // becuase the we don't know about the true label,
                && (votes.length >= (this.novelLabelIndex - 1)) // but we used a novel class detection classifier,
                && votes[this.novelLabelIndex] > 0) {           // and it marked it as a novel label
            isCorrect = true;                                   // Then we can count it as correct
            }
        }
        return isCorrect;
    }
    
    @Override
    public void trainOnInstance(Instance inst) {
        // By convention, instances with weight == 0 are for unsupervised training
        this.trainingWeightSeenByModel += inst.weight();
        trainOnInstanceImpl(inst);
    }
    
    final public Instance augmentInstance(Instance x) {
        Instance ret = (Instance) x.copy();
        ret.setDataset(augmentInstances(x.dataset()));
        
        return ret;
    }
    
    final public static Instances augmentInstances(Instances datum) {
        ArrayList<Attribute> attInfo = new ArrayList<>(datum.numAttributes());
        for(int aIdx = 0; aIdx < datum.numAttributes(); aIdx++) {
            Attribute a = datum.attribute(aIdx).copy(datum.attribute(aIdx).name());
            if ((aIdx == datum.classIndex()) && (a.indexOfValue(NOVEL_LABEL_STR) < 0)) { // only if we don't already have these
                List<String> values = new ArrayList<>(a.numValues() + 2);
                for(int i = 0; i < a.numValues(); ++i) {
                    values.add(a.value(i));
                }
                values.add(OUTLIER_LABEL_STR);
                values.add(NOVEL_LABEL_STR);
                a = new Attribute(a.name(),values,a.getMetadata());
            }
            attInfo.add(a);
        }
        String relationshipName = NOVEL_CLASS_INSTANCE_RELATIONSHIP_TYPE + "-" + datum.relationName();
        Instances ret = new Instances(relationshipName, attInfo, 1);
        ret.setClassIndex(datum.classIndex());
        
        return ret;
    }
}
