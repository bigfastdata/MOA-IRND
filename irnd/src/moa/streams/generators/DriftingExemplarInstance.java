/*
 *    InducedRandomNonStationaryDataGenerator.java
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
package moa.streams.generators;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeSet;
import weka.core.DenseInstance;
import weka.core.RevisionUtils;

/**
 * DriftingExemplarInstance.java
 *
 * Class for generating instances. This serves as the 'centroid' or exemplar instance, extending the SparseInstace. When
 * step() is called, the instance 'moves' through the feature space at some velocity, emulating concept drift.
 *
 * In addition, the attribute weights are modified slowly emulating feature drift.
 *
 * This class was originally designed for use by the RandomMixedNovelDriftGenerator for MOA as part of Brandon Parker's
 * Dissertation work.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 2 $
 */
public class DriftingExemplarInstance extends DenseInstance implements Comparable<DriftingExemplarInstance> {

    /**
     * for serialization
     */
    static final long serialVersionUID = 1482635194499365123L;
    protected int maxActive;
    protected double probAttributeActivation = 0;
    protected double probVelShift = 0;
    protected double variance = 0;
    protected int GMMSize = 1;
    protected ArrayList<DriftingExemplarAttribute> featureSet;
    protected double[] featureWeights;
    protected TreeSet<Integer> featuresEnabled;
    protected TreeSet<Integer> featuresDisabled;
    protected Random rng;
    /**
     * Constructor of an instance that sets weight to one, all values to be missing, and the reference to the dataset to
     * null. (ie. the instance doesn't have access to information about the attribute types)
     *
     * While this is heavy for a typical CTOR, it initializes values safely. Most of the work is 
     * in the DriftingExemplarAttribute class
     * 
     * @param p_featureSet Common set of features/Attributes
     * @param p_labelValue Label/class of the instance
     * @param p_maxActive sets the maximum number of active attributes for this Exemplar
     * @param p_probAttributeActivation robability that active attributes are toggled
     * @param p_maxVelocity Max average drift in attribute loci
     * @param p_probVelShift The probability that an attributes velocity will shift
     * @param p_variance Noise factor added to attribute values.
     * @param p_GMMSize The max number of centriods per attribute per label/class
     * @param p_rng Random number generator
     */
    // @ requires p_featureSet != null
    public DriftingExemplarInstance(
            ArrayList<DriftingExemplarAttribute> p_featureSet, 
            int p_labelValue,
            int p_maxActive, 
            double p_probAttributeActivation,
            double p_maxVelocity,
            double p_probVelShift,
            double p_variance,
            int p_GMMSize,
            Random p_rng) {
        super(p_featureSet.size());
        this.featureWeights = new double[p_featureSet.size()];
        for(int i = 0; i < this.featureWeights.length; ++i)    {
            this.featureWeights[i] = 1.0;
        }
        //this.featureWeights[featureWeights.length - 1] = 1.0;
        this.maxActive = Math.min(p_maxActive, this.featureWeights.length);
        this.probAttributeActivation = p_probAttributeActivation;
        this.featuresEnabled = new TreeSet<Integer>();
        this.featuresDisabled = new TreeSet<Integer>();
        this.featureSet = new ArrayList<DriftingExemplarAttribute>(p_featureSet.size());
        this.rng = p_rng;
        for (int i = 0; i < p_featureSet.size() - 1; ++i) {
            // Get a deep copy of the base attribute
            DriftingExemplarAttribute a = (DriftingExemplarAttribute) p_featureSet.get(i).copy();
            // Assign a GMM size and a max velocity based on a random value but bounded by the user parameters
            a.setGMM(Math.max(1, rng.nextInt(p_GMMSize) + 1), p_maxVelocity - (rng.nextDouble() * 0.66 * p_maxVelocity));
            // Keep p(velocityChange) exactly as the user requested
            a.setProbabilityOfVelocityShift(p_probVelShift);
            // Keep attribute value variance/noise exactly as the user requested
            a.setVariance(p_variance);
            // Add it to the containers needed
            featureSet.add(a);
            m_AttValues[i] = a.generateNextValue();
            this.featuresEnabled.add(i);
        }
        if (p_featureSet.size() > 0) {
            m_AttValues[p_featureSet.size() - 1] = p_labelValue;
            while(this.featuresEnabled.size() > Math.max(1,this.maxActive))
            {
                Integer[] activeArray = this.featuresEnabled.toArray(new Integer[1]);
                int idx = rng.nextInt(activeArray.length);
                this.featuresDisabled.add((int)activeArray[idx].longValue());
                this.featuresEnabled.remove((int)activeArray[idx].longValue());
                this.featureWeights[(int)activeArray[idx].longValue()] = 0.0;
            }
        }
    }

    /**
     *
     * @return new denseInstance from sample generator
     */
    public DenseInstance nextInstance() {
        DenseInstance ret = new DenseInstance(this);
        step();
        return ret;
    }

    /**
     * Drift the data and update the value array
     */
    public void step() {
        for (DriftingExemplarAttribute a :  featureSet) {
            int idx = a.attribute.index();
            m_AttValues[idx] = a.generateNextValue() * this.featureWeights[idx];
            a.step();
        }
        if (((this.maxActive + 1) < this.featureWeights.length) && (rng.nextDouble() < this.probAttributeActivation))
        {
            Integer[] activeArray = this.featuresEnabled.toArray(new Integer[1]);
            Integer[] inactiveArray = this.featuresDisabled.toArray(new Integer[1]);
            int idxToActivate = rng.nextInt(inactiveArray.length);
            int idxToDeactivate = rng.nextInt(activeArray.length);
            featuresEnabled.add(inactiveArray[idxToActivate]);
            featuresEnabled.remove(activeArray[idxToDeactivate]);
            featuresDisabled.add(activeArray[idxToDeactivate]);
            featuresDisabled.remove(inactiveArray[idxToActivate]);
            this.featureWeights[inactiveArray[idxToActivate]] = 1.0;
            this.featureWeights[activeArray[idxToDeactivate]] = 0.0;
        }
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }

    @Override
    public int compareTo(DriftingExemplarInstance other) {
        int ret = 0;
        if (this.classValue() < other.classValue()) {
            ret = -1;
        } else if (this.classValue() > other.classValue()) {
            ret = 1;
        } 
        return ret;
    }
}
