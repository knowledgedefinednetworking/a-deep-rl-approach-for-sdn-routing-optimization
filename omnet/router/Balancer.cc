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

#include "Balancer.h"


Define_Module(Balancer);


Balancer::Balancer() {
}

Balancer::~Balancer() {
}

void Balancer::initialize()
{
    numPorts = gateSize("in");
    id = par("id");
    numTx = par("numTx");
    numNodes = par("numNodes");
    folderName = par("folderName").stdstringValue();

    //for (int i = 0; i < numTx; i++) {
    //    balance[i] = uniform(0.05,0.95);
    //    Statistic::instance()->setRouting(id, i, balance[i]);
    //}

    // READ FROM FILE
    getBalancingInfo(id, balance);
    //generateBalancingInfo(id, balance);

    Statistic::instance()->setNumNodes(numNodes);
    Statistic::instance()->setNumTx(numTx);
    for (int dest = 0; dest < numTx; dest++) {
        Statistic::instance()->setRouting(id, dest, balance[dest]);
    }

}

void Balancer::handleMessage(cMessage *msg)
{
    DataPacket *data = check_and_cast<DataPacket *>(msg);

    if (id == data->getDstNode()) {
        ev << this->getFullPath() << "  Message received" << endl;
        simtime_t delayPaquet= simTime() - data->getCreationTime();
        Statistic::instance()->setDelay(simTime(), data->getSrcNode(), id, delayPaquet.dbl());
        delete msg;
    }
    else { // Tant in com out
        double aux = uniform(0,1);
        int destPort;
        if (aux < balance[data->getDstNode()])
            destPort = 0;
        else
            destPort = 1;

        send(msg, "out", destPort);

        ev << "Balancing: " << this->getFullPath() << "  Source: " << data->getSrcNode() << " Dest: " << data->getDstNode()
                << " using port: "<< destPort << endl;

    }
    //if (msg->arrivedOn("localIn")) {

}


void Balancer::getBalancingInfo(int id, double rData[]) {

     string line;
     ifstream myfile (folderName + "/Balancing.txt");
     double val;

     if (myfile.is_open()) {
         int i = 0;
         while (id != i) {
             for(int k = 0; k < numTx; k++) {
                 string aux;
                 getline(myfile, aux, ',');
             }
             //myfile >> val;
             i++;
         }

         for(int k = 0; k < numTx; k++) {
             string aux;
             getline(myfile, aux, ',');
             val = stod(aux);
             rData[k] = val;
         }

         myfile.close();
     }
}

void Balancer::generateBalancingInfo(int id, double rData[]) {

     for(int k = 0; k < numTx; k++) {
         rData[k] = uniform(0.05,0.95);;
     }
}
