#!/usr/bin/python
from classes.station.Station import Station
from classes.instrument.Instrument import Instrument
def main():
  sta = Station("HU", "BUD", 47, 19, 167)
  ins = Instrument(sta)
  ins.print_station()

  
if __name__== "__main__":
  main()
