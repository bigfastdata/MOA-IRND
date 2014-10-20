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

package moa.core;

import moa.options.MultiChoiceOption;
import weka.core.Attribute;
import weka.core.Instances;

/**
 *
 * @author bparker
 */
public class VectorDistances {

    public static final MultiChoiceOption distanceStrategyOption = new MultiChoiceOption("distanceStrategy", 'd',
            "Set strategy for distance measure.",
            new String[]{"Minimum",     // 0
                "Manhattan",            // 1
                "Euclidian",            // 2
                "Chebychev",            // 3
                "Aggarwal-0.1",         // 4
                "Aggarwal-0.3",         // 5
                "Average",              // 6
                "Chord",                // 7
                "Geo",                  // 8
                "Divergence",           // 9
                "Gower",                //10
                "Bray",                 //11
                "Jaccard",              //12
                "Probability"},         //13
            new String[]{"Minimum distance(L0 Norm)", 
                "Manhattan distance (L1 Norm), ", 
                "Euclidian distance (L2 Norm)", 
                "Chebychev distance (L-Inf Norm)", 
                "Aggarwal L-0.1 Norm (fractional minkowski power)",
                "Aggarwal L-0.3 Norm (fractional minkowski power)",
                "Average",
                "Chord",
                "Geo",
                "Divergence",
                "Gower",
                "Bray",
                "Jaccard",
                "P(x | c)"},
            13);
    
    public final static double distance(double[] src, double[] dst, Instances header, int typeIdx) {
        double distance = 0;
    // Choose distance algorithm
        switch (typeIdx) {
            case 0: // Minimum
                distance = VectorDistances.distanceMinkowski(src, dst, header, 0);
                break;
            case 1: // Manhattan
                distance = VectorDistances.distanceMinkowski(src, dst, header, 1);
                break;
            case 2: // Euclidian
                distance = VectorDistances.distanceMinkowski(src, dst, header, 2);
                break;
            case 3: // Chebyshev/Maximum
                distance = VectorDistances.distanceMinkowski(src, dst, header, Double.MAX_EXPONENT);
                break;
            case 4: // Aggarwal L-0.1
                distance = VectorDistances.distanceMinkowski(src, dst, header, 0.1);
                break;
            case 5: // Aggarwal L-0.3
                distance = VectorDistances.distanceMinkowski(src, dst, header, 0.3);
                break;
            case 6: // Average
                distance = VectorDistances.distanceAverage(src, dst, header);
                break;
            case 7: // Chord
                distance = VectorDistances.distanceChord(src, dst, header);
                break;
            case 8: // Geo
                distance = VectorDistances.distanceGeo(src, dst, header);
                break;
            case 9: // Divergance
                distance = VectorDistances.distanceDivergence(src, dst, header);
                break;
            case 10: // Gower
                // Gower does its own normalization, so use featureWeights directly here
                distance = VectorDistances.distanceGower(src, dst, header);
                break;
            case 11: // Bray
                distance = VectorDistances.distanceBray(src, dst, header);
                break;
            case 12: // Jaccard
                distance = VectorDistances.distanceJaccard(src, dst, header);
                break;
//            case 13: // Mahalanobis
//                distance = VectorDistances.distanceMahalanobis(src, dst, header);
//                break;
            default:
                distance = Double.MAX_VALUE;
        }
        return distance;
    }
    // Magic numbers
    private static final double epsilon = weka.core.Utils.SMALL;   // Default match envelope width is sigma not defined yet
    
    public static final double Minkowski_Manhattan = 1.0;
    public static final double Minkowski_Euclidean = 2.0;
    public static final double Minkowski_Chebyshev = Double.MAX_EXPONENT / 2.0;
    
    /**
     * Generalized Minkowski distance equation to cover the entire family of distances
     * power &lt; 1 --&gt; Minimum distance (strictly speaking, not a Minkowski distance)
     * power = 1 --&gt; Manhattan distance
     * power = 2 --&gt; Euclidian distance
     * power = INF --&gt; Chebyshev (or maximum) distance
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weight (strictly speaking, all weights should be 1 for pure Minkowski)
     * @param power power used to raise each component distance and 1/p for final reduction
     * @return Minkowski distance between the two points
     */
    public static synchronized double distanceMinkowski(double[] src, double[] dst, Instances header, double power) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        double minDist = Double.MAX_VALUE;
        double maxDist = Double.MIN_VALUE;
        for (int i = 0; i < minSize; i++) {
            double d = Math.abs(src[i] - dst[i]);
            double w = header.attribute(i).weight();
            ret += (d >= (epsilon*epsilon)) ? Math.abs(Math.pow(d, power)) * w : 0;
            if (w > 0) {minDist = Math.min(minDist, d);}
            if (w > 0) {maxDist = Math.max(maxDist, d);}
        }
        
        if (power >= Minkowski_Chebyshev) {
            ret = maxDist; 
        } else if (power < 0.000000001) {
            ret = minDist;
        } else {
            ret = (ret >= (epsilon * epsilon)) ? Math.pow(ret, 1.0 / power) : 0;
        }
        
        // Safety...
        if (Double.isInfinite(ret)) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
    
        return ret;
    }
    
    
    /**
     * Average distance, which is a modification of Euclidian distance
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weights and meta data (strictly speaking, all weights should be 1 for pure Minkowski)
     * @return component-averaged Euclidian distance
     */
    public static synchronized double distanceAverage(double[] src, double[] dst, Instances header) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        for (int i = 0; i < minSize; i++) {
            double d = Math.abs(src[i] - dst[i]);
            ret += d * d * header.attribute(i).weight();
        }
        ret = Math.sqrt(ret / minSize);
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }
    
    /**
     * Average distance, which is a modification of Euclidian distance
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header data set header used to determine attribute/feature type for mixed distance
     * @return component-averaged Euclidian distance
     */
    public static synchronized double distanceGower(double[] src, double[] dst, Instances header) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        double wSum = 0.0;
        for (int i = 0; i < minSize; i++) {
            Attribute att = header.attribute(i);
            double d = 0.0;
            double w = header.attribute(i).weight();
            if (att == null) { continue; }
            switch(att.type()) {
                case Attribute.NUMERIC:
                     w = (src[i] == 0 || dst[i] == 0) ? 0.0 : 1.0; 
                     double sigma = Math.abs(header.attribute(i).getUpperNumericBound() - header.attribute(i).getLowerNumericBound());
                     d = (Double.isFinite(sigma) && sigma > 0) ? Math.abs(src[i] - dst[i]) / sigma : Math.abs(src[i] - dst[i]) / 1;//Math.max(src[i], dst[i]);
                    break;
                case Attribute.NOMINAL:
                case Attribute.STRING:
                    d = (src[i] == dst[i]) ? 0.0 : 1.0; 
                    break;
                case Attribute.DATE:
                case Attribute.RELATIONAL:
                default:
                    System.err.println("Attribute type " + Attribute.typeToString(att) + " is not yet supported... ignoring feature "+ i);
                    d = 0.0;
                    w = 0;
            }
            wSum += w;
            ret += d * d * w;
        }
        ret = (wSum > 0) ? Math.sqrt(ret / wSum) : 0.0;
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }
    
    /**
     * Coefficient of Divergence (Legendre and Legendre, 1983)
     * Also known as the Canberra distance
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header  feature weight (strictly speaking, all weights should be 1 for pure Minkowski)
     * @return distance
     */
    public static synchronized double distanceDivergence(double[] src, double[] dst, Instances header) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        for (int i = 0; i < minSize; i++) {
            if (Math.abs(src[i] + dst[i]) <= 0) continue; 
            double d = Math.abs((src[i] - dst[i])/(src[i] + dst[i]));
            ret += d * d * header.attribute(i).weight();
        }
        ret = Math.sqrt(ret / minSize);
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }
    
    /**
     * Bray-Curtis distance
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weight and meta-data 
     * @return distance
     */
    public static synchronized double distanceBray(double[] src, double[] dst, Instances header) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        double numerator = 0;
        double denominator = 0;
        for (int i = 0; i < minSize; i++) {
            numerator += header.attribute(i).weight() * Math.abs(src[i] - dst[i]);
            denominator += header.attribute(i).weight() * Math.abs(src[i] + dst[i]);
        }
        ret += (denominator != 0.0) ? numerator / denominator : Double.MAX_VALUE;
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }
    
     /**
     * Jaccard index
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weight 
     * @return distance
     */
    public static synchronized double distanceJaccard(double[] src, double[] dst, Instances header) {
        double ret = distanceBray(src,dst,header);
        ret = 2 * ret / (1 + ret);
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }
    
    /**
     * Chord distance (Orloci, 1967)
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weight (strictly speaking, all weights should be 1 for pure Minkowski)
     * @return distance of the chord joining two normalized points within a hypersphere of radius 1
     */
    public static synchronized double distanceChord(double[] src, double[] dst, Instances header) {
        double ret = 0.0;
        int minSize = Math.min(src.length, Math.min(dst.length, header.numAttributes()));
        if (minSize < 1) { return Double.MAX_VALUE; }
        
        double srcL2Norm = 0.0;
        double dstL2Norm = 0.0;
        for (int i = 0; i < src.length; i++) {
            srcL2Norm += src[i] * src[i];
        }
        srcL2Norm = Math.sqrt(srcL2Norm);
        for (int i = 0; i < dst.length; i++) {
            dstL2Norm += dst[i] * dst[i];
        }
        dstL2Norm = Math.sqrt(dstL2Norm);
        
        for (int i = 0; i < minSize; i++) {
            ret += src[i] * dst[i] * header.attribute(i).weight();
        }
        ret = Math.abs(2.0 - 2 * (ret / (srcL2Norm * dstL2Norm)));
        ret = Math.sqrt(ret);
        
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }    
    
    
     /**
     * Geodesic distance (Legendre and Legendre, 1983)
     * @param src first data point to compare from
     * @param dst second data point to compare to
     * @param header feature weight (strictly speaking, all weights should be 1 for pure Minkowski)
     * @return distance of the chord joining two normalized points within a hypersphere of radius 1
     */
    public static synchronized double distanceGeo(double[] src, double[] dst, Instances header) {
        double ret = Math.acos(1.0 - distanceChord(src, dst, header) / 2.0);
        // Safety...
        if (Double.isInfinite(ret )) {
            ret = Double.MAX_VALUE;
        } else if (Double.isNaN(ret)) {
            ret = 0.0;
        } 
        return ret;
    }    
    
    /**
     * 
     * @param mean1 Gaussian Mean 1
     * @param var1 Gaussian variance 1
     * @param mean2 second average
     * @param var2 second variance
     * @return KL Divergence of the two Gaussian distributions (single dimensional Gaussian)
     */
    public static synchronized double KLDiverganceGaussian(double mean1, double var1, double mean2, double var2) {
        double term1 = Math.log(var2 / var1);
        double term2 = var1 / var2;
        double term3 = (mean2 - mean1) * (mean2 - mean1) / var2;
        if (Double.isNaN(term1)) {
            term1 = 0;
        }
        if (Double.isNaN(term2)) {
            term2 = 0;
        }
        if (Double.isNaN(term3)) {
            term3 = 0;
        }
        double ret = 0.5 * (term1 + term2 + term3 - 1);
        if (Double.isNaN(ret) || Double.isInfinite(ret)) {
            ret = 1.0 / weka.core.Utils.SMALL;
        }
        return ret;
    }
}
