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

#include "NodeQueue.h"

Define_Module(NodeQueue);

NodeQueue::NodeQueue() {


}

NodeQueue::~NodeQueue() {
    cancelAndDelete(endTxMsg);
    while (not portQueue.empty()) {
        delete portQueue.front();
        portQueue.pop();
    }

}

void NodeQueue::initialize()
{
    deleted = 0;
    endTxMsg = new cMessage("endTxMsg");

/*    string name = this->getFullPath();
    int r = 0;
    if (name[8] == 'r') r = 2;
    char c2 = name[13+r]; char c1 = name[12+r]; char p1 = name[20+r]; char p2 = name[21+r];
    if (c2 >= '0' and c2 <= '9') {
        idNode = (c1-'0')*10 + (c2 - '0');
        idPort = p2 - '0';
    }
    else {
        idNode = (c1-'0');
        idPort = p1 - '0';
    }*/
}

void NodeQueue::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        cMessage *packet = portQueue.front();
        portQueue.pop();
        send(packet, "line$o");
        if (not portQueue.empty()) {
            cChannel *txChannel = gate("line$o")->getTransmissionChannel();
            simtime_t txFinishTime = txChannel->getTransmissionFinishTime();
            scheduleAt(txFinishTime, endTxMsg);
        }

    }
    else if (msg->arrivedOn("in")) {
        //DataPacket *data = check_and_cast<DataPacket *>(msg);


        cChannel *txChannel = gate("line$o")->getTransmissionChannel();
        simtime_t txFinishTime = txChannel->getTransmissionFinishTime();
        if (txFinishTime <= simTime()) {
            // channel free; send out packet immediately
            send(msg, "line$o");
        }
        else {
            // store packet and schedule timer; when the timer expires,
            // the packet should be removed from the queue and sent out
            if (portQueue.empty())
                scheduleAt(txFinishTime, endTxMsg);
            if (portQueue.size() < 32)
                portQueue.push(msg);
            else {
                deleted++;
                DataPacket *data = check_and_cast<DataPacket *>(msg);
                Statistic::instance()->setLost(simTime(), data->getSrcNode(), data->getDstNode());
                delete msg;
            }
        }
        //ev << "QUEUE INFO  " << this->getFullPath() << "-->  Queue elements: " << portQueue.size() << endl;
        //ev << this->getFullPath() << "-->  Queue elements: " << portQueue.size() << endl;


    }
    else {
        send(msg,"out");
    }




}
