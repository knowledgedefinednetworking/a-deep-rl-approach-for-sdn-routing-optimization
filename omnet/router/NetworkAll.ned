package networksimulator;

network NetworkAll
{
    parameters:
        int numNodes = 14;
        int numTx = 14;

    types:
        channel Channel extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 9 kbps;
        }

        submodules:
            node0: Server {
                id = 0;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node1: Server {
                id = 1;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node2: Server {
                id = 2;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node3: Server {
                id = 3;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node4: Server {
                id = 4;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node5: Server {
                id = 5;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[4];
            }
            node6: Server {
                id = 6;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[2];
            }
            node7: Server {
                id = 7;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node8: Server {
                id = 8;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node9: Server {
                id = 9;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[4];
            }
            node10: Server {
                id = 10;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[2];
            }
            node11: Server {
                id = 11;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node12: Server {
                id = 12;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }
            node13: Server {
                id = 13;
                numTx = numTx;
                numNodes = numNodes;
                gates:
                    port[3];
            }

    connections:
        node0.port[0] <--> Channel <--> node1.port[0];
        node0.port[1] <--> Channel <--> node2.port[0];
        node0.port[2] <--> Channel <--> node3.port[0];
        node1.port[1] <--> Channel <--> node2.port[1];
        node1.port[2] <--> Channel <--> node7.port[0];
        node2.port[2] <--> Channel <--> node5.port[0];
        node3.port[1] <--> Channel <--> node4.port[0];
        node3.port[2] <--> Channel <--> node8.port[0];
        node4.port[1] <--> Channel <--> node5.port[1];
        node4.port[2] <--> Channel <--> node6.port[0];
        node5.port[2] <--> Channel <--> node10.port[0];
        node5.port[3] <--> Channel <--> node12.port[0];
        node6.port[1] <--> Channel <--> node7.port[1];
        node7.port[2] <--> Channel <--> node9.port[0];
        node8.port[1] <--> Channel <--> node11.port[0];
        node8.port[2] <--> Channel <--> node13.port[0];
        node9.port[1] <--> Channel <--> node10.port[1];
        node9.port[2] <--> Channel <--> node11.port[1];
        node9.port[3] <--> Channel <--> node13.port[1];
        node11.port[2] <--> Channel <--> node12.port[1];
        node12.port[2] <--> Channel <--> node13.port[2];
}
