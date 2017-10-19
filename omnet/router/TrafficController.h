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

#ifndef __SCALEFREE_TRAFFICCONTROLLER_H_
#define __SCALEFREE_TRAFFICCONTROLLER_H_

#include <omnetpp.h>
#include "ControlPacket_m.h"
using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <string>

/**
 * TODO - Generated class
 */
class TrafficController : public cSimpleModule
{
    private:
        double nodeRatio;
        int numNodes;
        int id;
        double flowRatio[100];
        string  folderName;

    protected:
        virtual void initialize();
        virtual void handleMessage(cMessage *msg);
        void getTrafficInfo(int id, double rData[]);
};

#endif
