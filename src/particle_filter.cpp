/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

std::default_random_engine defaultRandomEngine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (is_initialized) 
  {        
    return;
  }  
  
  num_particles = 100;  // TODO: Set the number of particles
  
  double std_x 		= std[0];
  double std_y 		= std[1];
  double std_theta 	= std[2];

  // Normal distributions
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);

  // Generate particles with normal distribution with mean on GPS values.
  for (int i = 0; i < num_particles; ++i) 
  {
    Particle pe;
    
    pe.id 		= i;
    pe.x 		= dist_x(defaultRandomEngine);
    pe.y 		= dist_y(defaultRandomEngine);
    pe.theta 	= dist_theta(defaultRandomEngine);
    pe.weight 	= 1.0;
    
    particles.push_back(pe);
  }
  
  is_initialized = true;

  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (int i = 0; i < num_particles; i++) 
  {
    // Move
    if (fabs(yaw_rate) >= 0.00001) 
    {
      // updating x, y and the yaw angle when the yaw rate is not equal to zero
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else 
    {
      // updating x, y when the yaw rate is equal to zero
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
   
    // Normal distributions for sensor noise
    std::normal_distribution<double> disX(0, std_pos[0]);
    std::normal_distribution<double> disY(0, std_pos[1]);
    std::normal_distribution<double> angle_theta(0, std_pos[2]);
    
    // Add noise
    particles[i].x 		+= disX(defaultRandomEngine);
    particles[i].y 		+= disY(defaultRandomEngine);
    particles[i].theta 	+= angle_theta(defaultRandomEngine);
  }

  return;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) 
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  int szPredicted = predicted.size();
  int szObservations = observations.size();

  for (int i = 0; i < szObservations; i++) 
  {
    // each observation
    double dMinDist = std::numeric_limits<double>::max();

    int idMap = -1;

    for (int j = 0; j < szPredicted; j++) 
    {
      // each predicted landmark
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // find closest predicted landmark for observation.
      if (distance < dMinDist) 
      {
        dMinDist = distance;
        idMap = predicted[j].id;
      }
    }

    observations[i].id = idMap;
  }
  
  return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double stdRange 	= std_landmark[0];
  double stdBearing = std_landmark[1];
  
  for (int i = 0; i < num_particles; i++) 
  {
    // Each particle
    double x 		= particles[i].x;
    double y 		= particles[i].y;
    double theta 	= particles[i].theta;

    vector<LandmarkObs> vReferencLandmarks;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) 
    {
      // Each landmark in map
      float landmarkX 	= map_landmarks.landmark_list[j].x_f;
      float landmarkY 	= map_landmarks.landmark_list[j].y_f;
      int id 			= map_landmarks.landmark_list[j].id_i;
      
      double dX = x - landmarkX;
      double dY = y - landmarkY;

      // Filter candidate landmarks according with extention.
      if (fabs(dX) <= sensor_range && fabs(dY) <= sensor_range) 
      {
        vReferencLandmarks.push_back(LandmarkObs{id, landmarkX, landmarkY});
      }
    }
    
    vector<LandmarkObs> vTransLandmarks;
    for (int j = 0; j < observations.size(); j++) 
    {
      // Each observation
      // Transform the car's measurements from its local car coordinate system to the map's coordinate system using a Homogenous Transformation matrix rotation/translation.
      double tx = x + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
      double ty = y + sin(theta) * observations[j].x + cos(theta) * observations[j].y;
      vTransLandmarks.push_back(LandmarkObs{observations[j].id, tx, ty});
    }

    // Find closest landmrk in map for each one observation.
    // Each measurement will be associated with a landmark identifier, take the closest landmark to each transformed observation
    dataAssociation(vReferencLandmarks, vTransLandmarks);
    
    particles[i].weight = 1.0;

    // Calculate the weight value of the particle
    for (unsigned int j = 0; j < vTransLandmarks.size(); j++) 
    {
      double observationX = vTransLandmarks[j].x;
      double observationY = vTransLandmarks[j].y;
      int landmarkId    = vTransLandmarks[j].id;

      double landmarkX, landmarkY;
      int k = 0;
      int nlandmarks = vReferencLandmarks.size();
      bool found = false;
      
      // Get observational landmark
      while (!found && k < nlandmarks) 
      {
        if (vReferencLandmarks[k].id == landmarkId) 
        {
          found = true;
          landmarkX = vReferencLandmarks[k].x;
          landmarkY = vReferencLandmarks[k].y;
        }
        
        k++;
      }
      
      // Weight for this observation with multivariate Gaussian
      // Calculate each measurement's Multivariate-Gaussian probability density
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;
      double weight = (1 / (2 * M_PI * stdRange * stdBearing)) * exp(-(dX * dX / (2 * stdRange * stdRange) + (dY * dY / (2 * stdBearing * stdBearing))));
      
      // Product of this obersvation weight with total observations weight
      // To get the final weight just multiply all the calculated measurement probabilities together.
      if (weight == 0) 
      {
        particles[i].weight = particles[i].weight * 0.00001;
      } 
      else 
      {
        particles[i].weight = particles[i].weight * weight;
      }
    }
  }

  return;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  // Calculate max weight.
  vector<double> weights;
  double maxWeight = std::numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++) 
  {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > maxWeight) 
    {
      maxWeight = particles[i].weight;
    }
  }
  
  std::uniform_real_distribution<float> dist_float(0.0, maxWeight);
  std::uniform_real_distribution<float> dist_int(0.0, num_particles - 1);
  
  int index = dist_int(defaultRandomEngine);
  double beta = 0.0;
  vector<Particle> resampledParticles;
  
  // Resample
  for (int i = 0; i < num_particles; i++) 
  {
    beta += dist_float(defaultRandomEngine) * 2.0;
    while (beta > weights[index]) 
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    
    resampledParticles.push_back(particles[index]);
  }
  
  // update particles
  particles = resampledParticles;
  
  return;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  
  return s;
}