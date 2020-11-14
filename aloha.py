import simpy
import sys
import collections
import numpy as np
import random
from scipy.stats import poisson, expon

BIT_TIME = 1
TRANSIENT_PERIOD = 25
TERMINATION_TIME = 10000

class Event(object):
  # event_type 0 == Arrival, event_type 1 == Departure, event_type 2 == Reset
  def __init__(self, event_type, event_time):
    self.event_type = event_type
    self.event_time = event_time

class Frame(object):
  def __init__(self, frame_size, arrival_time, station_id):
    self.frame_size = frame_size
    self.arrival_time = arrival_time
    self.station_id = station_id
    self.transmit_start = self.transmit_end = -1
    self.retransmit_frame = False

class Station(object):
  # Static dictionary of all the frames that each station will want to put on the medium.
  frames_on_medium = {}
  
  def __init__(self, env, poisson_mean, exponential_mean, station_id):
    self.clock = self.num_frames = self.num_retransmits = self.num_frames_transmitted = self.throughput = self.successful_utilization = 0
    self.num_waiting_times = self.waiting_time_sum = self.time_in_sys = self.last_event_time = self.total_transmit_times = 0
    self.total_service_time = self.service_duration = self.busy_start = self.total_time_busy = self.status = self.time_without_collision = 0
    self.mean_waiting_time = self.utilization = self.mean_transmit_time = self.mean_service_time = 0
    self.station_id = station_id
    self.env = env
    self.poisson_mean = poisson_mean
    self.exponential_mean = exponential_mean
    self.events_queue = collections.deque()
    self.frames_queue = collections.deque()

    # Scheduling the initial arrival.
    inter_t = poisson.rvs(self.poisson_mean)
    self.add_to_events_queue(Event(0, inter_t))

    # Scheduling the reset event.   
    self.add_to_events_queue(Event(2, (self.clock + TRANSIENT_PERIOD)))

    # Trigger the main simulation loop.
    self.env.process(self.run())    

  def run(self):
    while self.clock < TERMINATION_TIME:
      try:
        event = self.events_queue.popleft()
        self.clock = event.event_time
        self.time_in_sys += (self.clock - self.last_event_time) * self.num_frames
        self.last_event_time = self.clock

        if event.event_type == 0:
          self.handle_arrival()
          if self.status == 0:
            yield self.env.process(self.start_service())

        elif event.event_type == 1:
          self.handle_departure()
          if self.num_frames > 0:
            yield self.env.process(self.start_service())

        elif event.event_type == 2:
          self.handle_reset() 

      except Exception as e:
        print(e)

  def handle_arrival(self):
    inter_t = poisson.rvs(self.poisson_mean)
    self.add_to_events_queue(Event(0, (self.clock + inter_t)))
    self.num_frames += 1
    frame_size = self.get_frame_size()
    self.frames_queue.append(Frame(frame_size, self.clock, self.station_id))

  def handle_departure(self):
    self.num_frames -= 1
    self.status = 0
    self.total_time_busy += (self.clock - self.busy_start)
      
  def handle_reset(self):
    self.total_time_busy = 0
    self.time_in_sys = 0
    self.total_service_time = 0
    self.num_waiting_times = 0
    self.waiting_time_sum = 0
    self.num_retransmits = 0
    self.num_frames_transmitted = 0
    self.total_transmit_times = 0
    self.time_without_collision = 0

  def start_service(self):
    frame = self.frames_queue.popleft()
    self.num_waiting_times += 1
    self.waiting_time_sum += (self.clock - frame.arrival_time)
    self.status = 1
    self.busy_start = self.clock
    self.service_duration = 0
    service_time = self.clock - frame.arrival_time
    transmit_time_start = self.clock      

    # Waits for a frame to be successfully transmitted.
    yield self.env.process(self.attempt_transmit(frame))
    self.num_frames_transmitted += 1
    service_time += self.service_duration
    self.total_transmit_times += self.service_duration
    self.total_service_time += service_time 
    self.add_to_events_queue(Event(1, (self.clock + service_time)))

  def attempt_transmit(self, frame):
    frame_transmitted = False
    frame_time = frame.frame_size * BIT_TIME
    transmit_time_start = self.clock
    frame.transmit_start = transmit_time_start
    frame.transmit_end = transmit_time_start + frame_time
    collision_detected = False

    while not frame_transmitted:
      # Adding the current frame to the shared medium.
      Station.frames_on_medium[self.station_id] = frame
      num_collisions = 0

      # Yielding timeout 0 for the purpose of turning this function into a generator.
      yield self.env.timeout(0)
      self.service_duration += frame_time
      
      # Checking whether there are collisions on the medium.
      for sid in Station.frames_on_medium.keys():
        # sid = The station id of a frame in transit.
        if sid != self.station_id:
          frame_on_medium_start = Station.frames_on_medium[sid].transmit_start
          frame_on_medium_end = Station.frames_on_medium[sid].transmit_end

          if(frame.transmit_start < frame_on_medium_end and frame.transmit_end > frame_on_medium_start):
            Station.frames_on_medium[sid].retransmit_frame = True
            frame.retransmit_frame = True
            collision_detected = True
            num_collisions += 1

      if not collision_detected:
        self.time_without_collision += frame_time    

      Station.frames_on_medium.pop(self.station_id)     

      if frame.retransmit_frame:
        back_off_delay = poisson.rvs((num_collisions * 3 * frame_time))
        self.num_retransmits += 1
        frame.transmit_start += back_off_delay
        frame.transmit_end = frame.transmit_start + frame_time
        frame.retransmit_frame = False

        # This is to make up for the situation where another station sets my frame retransmit value to True
        # after I have already added 'frame_time' to the 'time_without_collision'.
        if not collision_detected:
          self.time_without_collision -= frame_time
      else:
        frame_transmitted = True

  def get_frame_size(self):
    getting_frame_size = True
    while getting_frame_size:
      val = expon.rvs(self.exponential_mean)
      val = int(val)
      if val != 0:
        getting_frame_size = False  
    return val

  # This fucntion adds events_queue and ensures that events in the queue are ordered by time
  def add_to_events_queue(self, event):
    size = len(self.events_queue)
    if size == 0:
      self.events_queue.append(event)
    else:
      inserted = False
      i = 0
      while not inserted:
        if i != size:
          if event.event_time < self.events_queue[i].event_time:
            # The following operation only works with Python 3.5 and above so make sure that is what you're using to run this.
            try:
              self.events_queue.insert(i, event)
              inserted = True
            except:
              print('AN ERROR WAS ENCOUNTERED! Make sure you are running the script with python 3.5 or above!')
              sys.exit()
        else:
          self.events_queue.append(event)
          inserted = True
        i += 1

  def generate_report(self):
    self.mean_waiting_time = (self.waiting_time_sum/self.num_waiting_times) if self.num_waiting_times != 0 else 0
    self.utilization = self.total_time_busy/(self.clock - TRANSIENT_PERIOD)
    self.successful_utilization = self.time_without_collision/(self.clock - TRANSIENT_PERIOD)
    self.mean_transmit_time = self.total_transmit_times/self.num_frames_transmitted if self.num_frames_transmitted != 0 else 0
    self.mean_service_time = self.total_service_time/self.num_frames_transmitted if self.num_frames_transmitted != 0 else 0
    self.throughput = self.num_frames_transmitted/self.clock


if __name__ == "__main__":
  poisson_mean = 10
  exponential_mean = 0.5
  num_stations = 8
  num_replications = 1000
  total_mean_transmit_times = []
  total_mean_retransmits = []
  total_utilization = []
  total_successful_utilization = []
  total_mean_waiting_times = []
  total_mean_service_time = []
  total_mean_throughput = [] 
  total_mean_frames_transmitted = []
  
  for i in range(num_replications):
    env = simpy.Environment()
    stations = [Station(env, poisson_mean, exponential_mean, station_id) for station_id in range(num_stations)]
    env.run(until = TERMINATION_TIME)

    # Genrating Reports for single replication.
    mean_transmit_times = 0
    mean_retransmits = 0
    utilization = 0
    successful_utilization = 0
    mean_waiting_times = 0
    mean_service_time = 0
    mean_throughput = 0
    mean_frames_transmitted = 0

    for station in stations: 
      station.generate_report()
      mean_transmit_times += station.mean_transmit_time
      mean_retransmits += station.num_retransmits
      utilization += station.utilization
      successful_utilization += station.successful_utilization
      mean_waiting_times += station.mean_waiting_time
      mean_service_time += station.mean_service_time
      mean_throughput += station.throughput
      mean_frames_transmitted += station.num_frames_transmitted
    
    total_mean_transmit_times.append(mean_transmit_times/num_stations)
    total_mean_retransmits.append(mean_retransmits/num_stations)
    total_utilization.append(utilization/num_stations)
    total_successful_utilization.append(successful_utilization/num_stations)
    total_mean_waiting_times.append(mean_waiting_times/num_stations) 
    total_mean_service_time.append(mean_service_time/num_stations)
    total_mean_throughput.append(mean_throughput/num_stations)
    total_mean_frames_transmitted.append(mean_frames_transmitted/num_stations)
  
  print()
  print('NUM STATIONS: %d, NUM REPLICATIONS: %d' % (num_stations, num_replications))
  print('POISSON MEAN: %d, EXPONENTIAL MEAN: %.1f' % (poisson_mean, exponential_mean))
  print('---------------------------------------------')
  print('Mean Throughput: %.3f frames per unit time' % (np.mean(total_mean_throughput)))
  print('Successful Channel Utilization: %.3f%%' % (np.mean(total_successful_utilization) * 100))
  print('Total Channel Utilization: %.3f%%' % (np.mean(total_utilization) * 100))
  print()
  print('Mean of Frames Transmitted: %.3f' % (np.mean(total_mean_frames_transmitted)))
  print('Mean Number Retransmits: %.3f' % (np.mean(total_mean_retransmits)))
  print()
  print('Mean Total Waiting Time: %.2f' % (np.mean(total_mean_waiting_times)))
  print('Mean Service Time: %.2f' % (np.mean(total_mean_service_time)))
  print('Mean Transmit Time: %.3f' % (np.mean(total_mean_transmit_times)))
  print('---------------------------------------------')
  print()