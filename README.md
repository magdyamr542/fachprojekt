## FachProjekt code

### Create virtual env and install reqs

1. `virtualenv --python=$(which python3) venv`
1. `pip3 install -r requirements.txt`

### Connect to the notebook server

1. Link [here](http://apollo.cs.tu-dortmund.de/fachprojekte)
   1. if not at the Uni you need a vpn connection.
   1. install a vpn client `sudo apt install -y openconnect`
   1. connect with `sudo openconnect --juniper stud2.vpn.tu-dortmund.de`
   1. type in your uni account `smamm***` and password `****`
   1. use the credentials sent to you in the Uni web mail box to access the server
