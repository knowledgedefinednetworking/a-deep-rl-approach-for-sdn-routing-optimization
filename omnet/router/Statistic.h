//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#ifndef STATISTIC_H_

#define STATISTIC_H_


#include "string.h"
#include <omnetpp.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;


class Statistic {

  public:
    static Statistic *instance();
    void infoTS(simtime_t time);
    void setDelay(simtime_t time, int i, int j, double d);
    void setTraffic(simtime_t time, int i, int j, double t);
    void setRouting(int n, int r, double p);
    void setLambda(double l);
    void setGeneration(int genType);
    void setLost(simtime_t time, int n, int p);
    void setLost(simtime_t time);
    void setNumTx(int n);
    void setNumNodes(int n);
    void setRoutingParaam(double r);
    void setMaxSim(double r);
    void setFolder(string folder);


    void printStats();

  protected:


    Statistic();

    virtual ~Statistic();


  private:
    static Statistic *inst;
    simtime_t INI;
    simtime_t END;
    double SIMTIME;
    bool collect;

    int numTx;
    int numNodes;
    int genT;
    double lambdaMax;
    double routingP;

    string folderName;


    int drops;
    vector< vector< vector <double> > > Traffic;
    vector< vector <double> > Routing;
    vector< vector< vector <double> > >  Delay;
    vector< vector <double> >   DropsV;

    void initLinkID();

};



#endif /* STATISTIC_H_ */

