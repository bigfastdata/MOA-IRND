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

import java.io.Serializable;
import java.util.List;
import java.util.Random;
import weka.core.Attribute;
import weka.core.Copyable;
import weka.core.RevisionHandler;

/**
 * DriftingExemplarAttribute.java
 *
 * Extends the WEKA Attribute class to provide for a drifting attribute. The attribute weight and value drift whenever
 * step() is called. In addition, the value is selected from a GMM to better emulate real-world data that is not necessarily
 * best described by a single Gaussian distribution.
 *
 * This class was originally designed for use by the RandomMixedNovelDriftGenerator for MOA as part of Brandon Parker's
 * Dissertation work.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class DriftingExemplarAttribute implements Copyable, Serializable, RevisionHandler {
    
    private static final long serialVersionUID = 1L;
    protected Attribute attribute;
    protected double variance = 0;
    protected double[] expectedValues;
    protected int gmm_selector = 0;
    protected double maxVelocity = 0;
    protected double[] velocity;
    protected double probabilityOfVelocityShift = 0;
    final protected boolean reflectOffBoundaries = true;
    Random rng;

    /**
     * Construct an empty attribute - used for internal copy/clone functions only, hence this method is private
     * @param p_rng common Random Number Generator to use
     */
    private DriftingExemplarAttribute(Random p_rng) {
        attribute = null;
        rng = p_rng;
        variance = rng.nextDouble() * 0.15; // Default to modest variance
        expectedValues = new double[1];//Math.max(1, rng.nextInt(2) + 1)]; // Between 1 and 3 models in mixture
        velocity = new double[expectedValues.length]; 
        for(int i = 0; i < expectedValues.length; i++) {
            velocity[i] = rng.nextDouble() * 0.5 - 0.25;
            expectedValues[i] = rng.nextDouble();
        }
    }
    
    /**
     * Construct a numeric attribute
     *
     * @param p_name Attribute name (will appear in ARFF file)
     * @param p_conceptId Concept label 
     * @param p_rng Shared Random Number Generator
     */
    public DriftingExemplarAttribute(String p_name, int p_conceptId, Random p_rng) {
        attribute = new Attribute(p_name, p_conceptId);
        rng = p_rng;
        variance = 0;//rng.nextDouble() * 0.15; // Default to modest variance
        expectedValues = new double[1];//Math.max(1, rng.nextInt(2) + 1)]; // Between 1 and 3 models in mixture
        velocity = new double[expectedValues.length]; 
        for(int i = 0; i < expectedValues.length; i++) {
            velocity[i] = 0;// rng.nextDouble() * 0.5 - 0.25;
            expectedValues[i] = rng.nextDouble();
        }
    }

    /**
     * Construct a nominal or string attribute
     *
     * @param p_name name of this feature
     * @param p_corpus bag of words to choose from for this concept
     * @param p_conceptId index of this feature
     * @param p_rng Shared Random Number Generator
     */
    public DriftingExemplarAttribute(String p_name, int p_conceptId, Random p_rng, List<String> p_corpus) {
        attribute = new Attribute(p_name, p_corpus, p_conceptId);
        rng = p_rng;
        variance = 0;//rng.nextDouble() * 0.15; // Default to modest variance
        expectedValues = new double[Math.max(1, rng.nextInt(2) + 1)]; // Between 1 and 3 models in mixture
        velocity = new double[expectedValues.length]; 
        for(int i = 0; i < expectedValues.length; i++) {
            velocity[i] = 0;//rng.nextDouble() * 0.5 - 0.25;
            expectedValues[i] = rng.nextDouble();
        }
    }

    /**
     * Produces a shallow copy of this attribute and deep copy of other aspects.
     * @return a copy of this attribute with the same index
     */
    //@ also ensures \result instanceof Attribute;
    @Override
    public /*@ pure non_null @*/ Object copy() {
        DriftingExemplarAttribute copied = new DriftingExemplarAttribute(this.rng);
        copied.attribute        = (Attribute) this.attribute.copy();        
        copied.variance         = this.variance;
        copied.gmm_selector     = this.gmm_selector;
        copied.maxVelocity      = this.maxVelocity;
        copied.rng              = this.rng;
        copied.probabilityOfVelocityShift = this.probabilityOfVelocityShift;
        copied.expectedValues = new double[this.expectedValues.length]; 
        copied.velocity = new double[copied.expectedValues.length]; 
        for(int i = 0; i < copied.expectedValues.length; i++) {
            copied.velocity[i] = this.velocity[i];
            copied.expectedValues[i] = this.expectedValues[i];
        }
        return copied;
    }

    /**
     * Incrementally move/drift this attribute based on velocity
     */
    public void step() {
        if (reflectOffBoundaries) // Reflective edges
        {
            expectedValues[gmm_selector] += velocity[gmm_selector];
            if (expectedValues[gmm_selector] < 0.0) {
                expectedValues[gmm_selector] = 0.000001;
                velocity[gmm_selector] *= -1.0;
            }
        if (expectedValues[gmm_selector] > 1.0) {
                expectedValues[gmm_selector] = 0.999999;
                velocity[gmm_selector] *= -1.0;
            }
        }
        else { // Taurus edge
            expectedValues[gmm_selector] += velocity[gmm_selector] % 1.0;
        }
        
        if (rng.nextDouble() < this.probabilityOfVelocityShift)
        {
            for(int i = 0; i < velocity.length; i++) {
                velocity[i] = rng.nextDouble() * 2 * this.maxVelocity - this.maxVelocity;
            }
        }
    }
    
  
    /**
     * Sample next attribute value
     *
     * @return value of attribute for next instance
     */
    public double generateNextValue() {
        gmm_selector = this.rng.nextInt(expectedValues.length);
        double ret = expectedValues[gmm_selector] + this.rng.nextGaussian() * variance;
        if (ret < 0) { ret = 0; }
        if (ret >= 1) { ret = 0.99999; }
        if (this.attribute.isNominal() || this.attribute.isString()) {
            ret = Math.floor(this.attribute.numValues() * ret);
        }
        return ret;
    }

    /**
     * assign attribute
     * @param p_attrib attribute src
     */
    public void setAttribute(Attribute p_attrib) {
        attribute = p_attrib;
    }

    public void setProbabilityOfVelocityShift( double p_prob) {
        this.probabilityOfVelocityShift = p_prob;
    }
    
    /**
     * setter function
     *
     * @param p_variance Gaussian variance to set for attribute for generation of data values
     */
    public void setVariance(double p_variance) {
        this.variance = p_variance;
    }

    /**
     * setter function
     *
     * @param p_velocity value to set
     */
    public void setVelocity(double p_velocity) {
        this.maxVelocity = p_velocity;
        for(int i = 0; i < velocity.length; i++) {
            this.velocity[i] = (this.rng.nextDouble() * 2 * this.maxVelocity) - this.maxVelocity;
        }
    }
    
    /**
     * setter function
     *
     * @param p_velocity value to set
     */
    public void setVelocity(double p_velocity[]) {
        this.velocity = p_velocity;
    }

    /**
     * setter function
     *
     * @param p_gmmSize Size of the Gaussian Mixture model (i.e. number of centroids) to use
     * @param p_velocity value to set
     */
    public void setGMM(int p_gmmSize, double p_velocity) {
        this.maxVelocity = p_velocity;
        this.expectedValues = new double[Math.max(1, p_gmmSize)];
        this.velocity = new double[expectedValues.length]; 
        for(int i = 0; i < this.velocity.length; i++) {
            this.velocity[i] = (this.rng.nextDouble() * 2 * this.maxVelocity) - this.maxVelocity;
            this.expectedValues[i] = this.rng.nextDouble();
        }
        gmm_selector = 0;
    }

    /**
     *
     * @return drift velocity for generator
     */
    public double[] getVelocity() {
        return velocity;
    }

    /**
     *
     * @return variance value for generator
     */
    public double getVariance() {
        return variance;
    }

    /**
     * Gets the GMM size
     *
     * @return size of Gaussian Mixture Model
     */
    public int getGMMSize() {
        return this.expectedValues.length;
    }

    /**
     * Get attribute object
     * @return attribute (Weka) object
     */
    public Attribute getAttribute(){
        return attribute;
    }
    
    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
