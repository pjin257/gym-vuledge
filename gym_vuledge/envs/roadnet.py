import osmnx as ox
import networkx as nx
import numpy as np
import random
import simpy
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

class GV:
    SIM_TIME = 500 + 1
    GEN_RATE = 10
    GEN_END = (SIM_TIME - 1) * GEN_RATE * 0.9
    ATK_RATE = 100

    MOVE_INTV = 0.5
    
    REROUTE = None
    REROUTE_INTV = [60, 60]
    ROUTE_GRP_NUM = 2

    VEHICLE_LENGTH = 4.5
    WARMING_UP = 100

    BETA_1 = 1
    BETA_2 = 3
    BETA_3 = 20
    CP_1 = 0.5
    CP_2 = 0.8
    NEXT_CP = 0.5
    
    links = [
        'Motorway_link', 'Primary_link',
        'Secondary_link', 'Tertiary_link'
    ]

# Traffic generation process
class Traffic_Gen(object): 
    def __init__(self, env, G):
        self.env = env
        self.gen_rate = GV.GEN_RATE
        self.vehicle_number = 0
        self.G = G
        self.nodes = list(self.G.nodes)
        self.route_groups = list(range(0, GV.ROUTE_GRP_NUM))

        self.Q_dic = {}   # Dictionary of traffic queue for each edge in the network 
        self.delay_dic = {}   # Dictionary of practical delay log on each edge
        
        for edge in self.G.edges:
            self.Q_dic[edge] = []
            self.delay_dic[edge] = []

        self.action = env.process(self.run())
        
    def run(self):       
        while True:
            # Infinite loop for generating traffic
            yield self.env.timeout(random.expovariate(self.gen_rate))
                           
            if self.vehicle_number >= GV.GEN_END : continue
            
            # Create and enqueue new vehicle
            gen_time = self.env.now  
            
            generated = False
            gen_trial = 0
            
            while not generated:
                try :
                    # Generate a vehicle with random src/dst pair
                    src = random.choice(self.nodes)
                    dst = random.choice(self.nodes)
                    
                    if src == dst:
                        raise ValueError('src and dst node are the same')

                    cost = nx.shortest_path_length(self.G, src, dst, weight='expected_delay')
                    if cost == float('inf'):
                        raise ValueError('The only path for the OD pair is attacked')    
                        
                    path = nx.shortest_path(self.G, src, dst, weight='expected_delay')
                    start_edge = (path[0], path[1], 0)
                    if self.G.edges[start_edge]['saturation'] > 1:
                        raise ValueError('The start edge is fully saturated')

                    grp = random.choice(self.route_groups)
                    new_vehicle = Vehicle(self.vehicle_number, self.env.now, src, dst, path, grp)

                    # Put the vehicle in the starting edge
                    self.vehicle_entry(start_edge, new_vehicle)
                    self.G.edges[start_edge]['edge_cnt'] += 1
                    
                    self.vehicle_number += 1
                    generated = True

                except Exception as error:
                    # Errors above and there are some pairs that have no path between src/dst
                    print(error)
                    pass
            
    def vehicle_entry(self, edge, vehicle):
        # Add vehicle to the queue of selected edge
        q = self.Q_dic[edge]
        q.append(vehicle)
        vehicle.entry_time = self.env.now
        vehicle.edge_delay = self.G.edges[edge]['total_delay']
        vehicle.edge_sat = self.G.edges[edge]['saturation']
        
        # Update peak traffic and saturation rate of the edge
        trf_len = len(q)
        if trf_len > self.G.edges[edge]['peak_traffic']:
            self.G.edges[edge]['peak_traffic'] = trf_len
        self.G.edges[edge]['saturation'] = (trf_len * GV.VEHICLE_LENGTH) / (self.G.edges[edge]['length'] * self.G.edges[edge]['lanes'])
        
        # Update edge delay
        self.update_delay(edge)
    
    def update_delay(self, edge):
        base_delay = self.G.edges[edge]['travel_time']
        signal_delay = 0
        congest_delay = 0

        edge_type = self.G.edges[edge]['highway']
        saturation = self.G.edges[edge]['saturation']
        edge_len = self.G.edges[edge]['length']
        beta_1 = GV.BETA_1
        beta_2 = GV.BETA_2
        beta_3 = GV.BETA_3
        cp1 = GV.CP_1
        cp2 = GV.CP_2

        if edge_type == 'primary' or edge_type == 'primary_link':
            if edge_len > 15 : signal_delay = 10

        elif edge_type == 'secondary' or edge_type == 'secondary_link':
            if edge_len > 15 : signal_delay = 10

        elif edge_type == 'tertiary' or edge_type == 'tertiary_link':
            if edge_len > 15 : signal_delay = 6

        # Get penalty rate    

        # Is any of possible next edges congested?
        next_edge_congested = False
        (current_node, next_node, key) = edge

        for (u, v) in self.G.out_edges(next_node):
            if v != current_node:
                if self.G.edges[(u, v, 0)]['saturation'] > GV.NEXT_CP and self.G.edges[(u, v, 0)]['highway'] not in GV.links:
                    next_edge_congested = True

        if saturation < cp1:
            penalty_rate = beta_1 * saturation
        elif saturation >= cp1 and saturation < cp2:
            penalty_rate = cp1 * beta_1 + (saturation - cp1) * beta_2
        elif saturation >= cp2 and saturation <= 1:
            penalty_rate = cp1 * beta_1 + (cp2 - cp1) * beta_2 + (saturation - cp2) * beta_3
        else:
            penalty_rate = cp1 * beta_1 + (cp2 - cp1) * beta_2 + (1 - cp2) * beta_3

        # Get congestion delay
        congest_delay = penalty_rate * base_delay

        # Get total delay and add it to the dictionary
        delay_sum = base_delay + signal_delay + congest_delay
        self.G.edges[edge]['total_delay'] = delay_sum
        
        if self.G.edges[edge]['alive'] == True:
            # Get expected delay that approximate D_t
            if next_edge_congested:
                if saturation < cp1:
                    exp_penalty_rate = beta_1 * saturation
                elif saturation >= cp1 and saturation <= 1:
                    exp_penalty_rate = cp1 * beta_1 + (saturation - cp1) * beta_3
                else:
                    exp_penalty_rate = cp1 * beta_1 + (1 - cp1) * beta_3                    
                exp_congest_delay = exp_penalty_rate * base_delay
                exp_delay_sum = base_delay + signal_delay + exp_congest_delay

                self.G.edges[edge]['expected_delay'] = exp_delay_sum
            else:
                self.G.edges[edge]['expected_delay'] = delay_sum
        else:
            self.G.edges[edge]['expected_delay'] = float('inf')
            
# Vehicle instance
class Vehicle:
    def __init__(self, identifier, gen_time, src, dst, path, route_group):
        # Basic stats
        self.identifier = identifier
        self.gen_time = gen_time
        self.src = src
        self.dst = dst
        self.grp = route_group
        
        # dynamic stats
        self.path = path
        self.e_idx = 0
        self.entry_time = gen_time
        self.wait_time = 0 
        self.edge_delay = None
        self.edge_sat = None
        self.arrival_time = None
        self.trapped = False

# Moving process
class Moving_Process(object): 
    def __init__(self, env, G, traffic_generator):
        self.env = env
        self.interval = GV.MOVE_INTV
        self.finished = []
        self.G = G   
        self.tg = traffic_generator

        # Refine the number of lanes of each edge
        double_lane = ['East Covell Boulevard', 'West Covell Boulevard',
                      'Covell Boulevard', 'Russell Boulevard',
                      'East Covell Boulevard;West Covell Boulevard']
        triple_lane = []
        lane_dic = {}

        for edge in self.G.edges:
            # Get the number of lane
            edge_data = self.G.get_edge_data(edge[0], edge[1], edge[2])
            num_lane = edge_data.get('lanes')

            if isinstance(num_lane, list):
                num_lane = int(max(num_lane))
            elif isinstance(num_lane, str):
                num_lane = int(num_lane)
            elif num_lane == None: 
                num_lane = 1


            # Find the edges that have designated number of lanes
            road_name = edge_data.get('name')
            if isinstance(road_name, list):
                road_name = road_name[0]

            tem_lane_num = 0
            if road_name in double_lane :
                tem_lane_num = 2
            elif road_name in triple_lane :
                tem_lane_num = 3
            else : tem_lane_num = 1

            final_num = max(num_lane, tem_lane_num)
            lane_dic[edge] = final_num

        nx.set_edge_attributes(self.G, lane_dic, 'lanes')

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.interval)

            # Update 'total_delay' attribute of all the edges
            for edge in self.G.edges:
                self.update_delay(edge)
            
            # Move vehicles

            # Get randomly ordered edges
            edges = list(self.G.edges)
            random_order = random.sample(edges, len(edges))

            for edge in random_order:
              q = self.tg.Q_dic[edge]
              stuck = False

              if len(q) > 0:
                for vehicle in q:
                    vehicle.wait_time += self.interval
                    current_edge = (vehicle.path[vehicle.e_idx], vehicle.path[vehicle.e_idx + 1], 0)

                    last_edge = False
                    try :
                        next_edge = (vehicle.path[vehicle.e_idx + 1], vehicle.path[vehicle.e_idx + 2], 0)
                        next_sat = self.G.edges[next_edge]['saturation']
                    except Exception:
                        last_edge = True
                    
                    if not stuck: # Is this vehicle at the top of the queue?
                        if vehicle.wait_time >= vehicle.edge_delay: # Did the vehicle on top of the queue finish travel the edge?
                            # Check if current edge is the last edge
                            if last_edge:
                                vehicle.arrival_time = self.env.now
                                self.finished.append(vehicle)
                                self.vehicle_exit(current_edge, None, vehicle, last_edge)
                            
                            # Move to next edge if it is not full
                            elif next_sat < 1:                                    
                                # finish traveling current edge
                                self.vehicle_exit(current_edge, next_edge, vehicle)
                                vehicle.e_idx += 1 
                                    
                                # put the vehicle into the next edge    
                                self.vehicle_entry(next_edge, vehicle)
                                self.G.edges[next_edge]['edge_cnt'] += 1
                                vehicle.wait_time = 0
                            else:
                                stuck = True
                        else:
                            stuck = True

            print("Simulation time: {0}".format(self.env.now))
            
    def vehicle_entry(self, edge, vehicle):
        # Add vehicle to the queue of selected edge
        q = self.tg.Q_dic[edge]
        q.append(vehicle)
        vehicle.entry_time = self.env.now
        vehicle.edge_delay = self.G.edges[edge]['total_delay']
        vehicle.edge_sat = self.G.edges[edge]['saturation']
        
        # Update peak traffic and saturation rate of the edge
        trf_len = len(q)
        if trf_len > self.G.edges[edge]['peak_traffic']:
            self.G.edges[edge]['peak_traffic'] = trf_len
        self.G.edges[edge]['saturation'] = (trf_len * GV.VEHICLE_LENGTH) / (self.G.edges[edge]['length'] * self.G.edges[edge]['lanes'])
        
        # Update edge delay
        self.update_delay(edge)

    def vehicle_exit(self, edge, next_edge, vehicle, last_edge=False):
        # Remove vehicle to the queue of selected edge
        q = self.tg.Q_dic[edge]
        q.remove(vehicle)

        # Update delay log of the exiting edge
        delay = self.env.now - vehicle.entry_time
        entry_sat = vehicle.edge_sat
        if next_edge is not None:
            next_sat = self.G.edges[next_edge]['saturation']
        else:
            next_sat = None
        log = (entry_sat, next_sat, delay)
        self.tg.delay_dic[edge].append(log)

        # Update saturation rate of the edge
        trf_len = len(q)
        self.G.edges[edge]['saturation'] = (trf_len * GV.VEHICLE_LENGTH) / (self.G.edges[edge]['length'] * self.G.edges[edge]['lanes'])
        
        # Update edge delay
        self.update_delay(edge)
    
    def update_delay(self, edge):
        base_delay = self.G.edges[edge]['travel_time']
        signal_delay = 0
        congest_delay = 0

        edge_type = self.G.edges[edge]['highway']
        saturation = self.G.edges[edge]['saturation']
        edge_len = self.G.edges[edge]['length']
        beta_1 = GV.BETA_1
        beta_2 = GV.BETA_2
        beta_3 = GV.BETA_3
        cp1 = GV.CP_1
        cp2 = GV.CP_2

        if edge_type == 'primary' or edge_type == 'primary_link':
            if edge_len > 15 : signal_delay = 10

        elif edge_type == 'secondary' or edge_type == 'secondary_link':
            if edge_len > 15 : signal_delay = 10

        elif edge_type == 'tertiary' or edge_type == 'tertiary_link':
            if edge_len > 15 : signal_delay = 6

        # Get penalty rate    

        # Is any of possible next edges congested?
        next_edge_congested = False
        (current_node, next_node, key) = edge

        for (u, v) in self.G.out_edges(next_node):
            if v != current_node:
                if self.G.edges[(u, v, 0)]['saturation'] > GV.NEXT_CP and self.G.edges[(u, v, 0)]['highway'] not in GV.links:
                    next_edge_congested = True

        if saturation < cp1:
            penalty_rate = beta_1 * saturation
        elif saturation >= cp1 and saturation < cp2:
            penalty_rate = cp1 * beta_1 + (saturation - cp1) * beta_2
        elif saturation >= cp2 and saturation <= 1:
            penalty_rate = cp1 * beta_1 + (cp2 - cp1) * beta_2 + (saturation - cp2) * beta_3
        else:
            penalty_rate = cp1 * beta_1 + (cp2 - cp1) * beta_2 + (1 - cp2) * beta_3

        # Get congestion delay
        congest_delay = penalty_rate * base_delay

        # Get total delay and add it to the dictionary
        delay_sum = base_delay + signal_delay + congest_delay
        self.G.edges[edge]['total_delay'] = delay_sum
        
        if self.G.edges[edge]['alive'] == True:
            # Get expected delay that approximate D_t
            if next_edge_congested:
                if saturation < cp1:
                    exp_penalty_rate = beta_1 * saturation
                elif saturation >= cp1 and saturation <= 1:
                    exp_penalty_rate = cp1 * beta_1 + (saturation - cp1) * beta_3
                else:
                    exp_penalty_rate = cp1 * beta_1 + (1 - cp1) * beta_3                    
                exp_congest_delay = exp_penalty_rate * base_delay
                exp_delay_sum = base_delay + signal_delay + exp_congest_delay

                self.G.edges[edge]['expected_delay'] = exp_delay_sum
            else:
                self.G.edges[edge]['expected_delay'] = delay_sum
        else:
            self.G.edges[edge]['expected_delay'] = float('inf')

# Rerouting process
class Reroute_Process(object): 
    def __init__(self, env, G, traffic_generator, route_group, intv):
        self.env = env
        self.interval = intv
        self.G = G   
        self.tg = traffic_generator
        self.grp = route_group
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.interval)

            for edge in self.G.edges:
              q = self.tg.Q_dic[edge]
              for vehicle in q:
                if vehicle.grp == self.grp:
                    # the index of the node that this vehicle is moving toward on this edge
                    next_node_idx = vehicle.e_idx + 1

                    next_node = vehicle.path[next_node_idx]
                    left_path = vehicle.path[next_node_idx:]
                    new_path = nx.shortest_path(self.G, next_node, vehicle.dst, weight='expected_delay')

                    if left_path != new_path: # reroute if there is a shorter path
                        history = vehicle.path[:next_node_idx]
                        new_route = history + new_path
                        vehicle.path = new_route
   
# Disruption on edges
class Edge_Attack(object): 
    def __init__(self, env, G, traffic_generator):
        self.env = env
        self.atk_rate = GV.ATK_RATE
        self.G = G
        self.hist = {}
        self.tg = traffic_generator
        self.atk_cnt = 0

        # manage candidates and select
        self.cand = {}
        self.disrupt_order = None
        self.last_atk_time = None

        # Term edge count after idx-th attack
        self.term_edge_cnt = []

        self.action = env.process(self.run())
        
    def run(self):       
        yield self.env.timeout(GV.WARMING_UP) # defer attack to the end of warming-up period
        while True:                        
            # Select an edge to disrupt that chosen by the agent
            for key in self.cand:
                if self.cand[key] == self.disrupt_order:
                    vul_edge = key
            
            # Change expected time cost to infinity
            self.G.edges[vul_edge]['expected_delay'] = float('inf')
            self.G.edges[vul_edge]['alive'] = False
            
            # Log attack history (edge, atk_time)
            self.hist[self.env.now] = vul_edge

            # Reset edge count to get count per attack rate in next iteration
            term_cnt = {}
            for edge in self.G.edges:
              term_cnt[edge] = self.G.edges[edge]['edge_cnt']
              self.G.edges[edge]['acc_edge_cnt'] += self.G.edges[edge]['edge_cnt']
            nx.set_edge_attributes(self.G, 0, 'edge_cnt')
            self.term_edge_cnt.append(term_cnt)
            
            # Reroute to avoid the disrupted edge, as later as possible before arriving the edge
            self.reroute(vul_edge)

            self.atk_cnt += 1
            self.last_atk_time = self.env.now

            yield self.env.timeout(self.atk_rate)
            
    def reroute(self, vul_edge): 
        disrupted_start = vul_edge[0]
        disrupted_end = vul_edge[1]
        
        for edge in self.G.edges:
            q = self.tg.Q_dic[edge]
            for vehicle in q:
                
                # Is this vehicle go through the disrupted edge?
                if disrupted_start in vehicle.path:
                    disrupted_idx = vehicle.path.index(disrupted_start)
                    next_node_idx = vehicle.e_idx + 1
                    
                    # not yet reached the disrupted edge?
                    if next_node_idx <= disrupted_idx:
                        
                        next_node = vehicle.path[next_node_idx]
                        cost2goal = nx.shortest_path_length(self.G, next_node, vehicle.dst, weight='expected_delay')
                        
                        # Is there any alternative path after disruption?
                        if cost2goal != float('inf'):
                        
                            # reroute at the nearest node from the disrupted edge in the path
                            for i in reversed(range(next_node_idx, disrupted_idx + 1)):
                                reroute_node_idx = i
                                reroute_node = vehicle.path[reroute_node_idx]
                                left_path = vehicle.path[reroute_node_idx:]
                                
                                try :
                                    new_path_cost = nx.shortest_path_length(self.G, reroute_node, vehicle.dst, weight='expected_delay')
                                    
                                    if new_path_cost < float('inf'):
                                        new_path = nx.shortest_path(self.G, reroute_node, vehicle.dst, weight='expected_delay')
                                        if left_path != new_path: # reroute if there is a shorter path
                                            history = vehicle.path[:reroute_node_idx]
                                            new_route = history + new_path
                                            vehicle.path = new_route
                                    
                                except Exception as error:
                                    print('[Re-routing]', error)

class ROADNET(object):
    def __init__(self, num_cand):
        # Define variables
        self.G = None
        self.env = None
        self.tg = None
        self.mv = None
        self.edge_atk = None
        self.rr = None
        self.NUM_CAND = num_cand
        self.ONEHOT_CAND = LabelBinarizer().fit_transform(np.arange(0, self.NUM_CAND, 1))

        self.init_graph()

    def init_graph(self):
        # Set boundaries and filters
        north, south, east, west = 38.5754, 38.5156, -121.6759, -121.7941
        rf = '["highway"~"motorway|motorway_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link"]'

        # load data
        G = ox.graph_from_bbox(north, south, east, west, network_type='drive', custom_filter=rf)
        
        ebunch = []

        for (u,v,k) in G.edges:  
            if k != 0:
                ebunch.append((u,v,k))
        G.remove_edges_from(ebunch)

        nx.set_edge_attributes(G, 0, 'peak_traffic')
        nx.set_edge_attributes(G, 0, 'saturation')
        nx.set_edge_attributes(G, 0, 'edge_cnt')
        nx.set_edge_attributes(G, 0, 'acc_edge_cnt')
        nx.set_edge_attributes(G, True, 'alive')

        # Set default speed of vehicles on each type of road
        hwy_speeds = {'motorway': 110,
                    'motorway_link': 60,
                    'primary': 60,
                    'primary_link': 50,
                    'secondary': 50,
                    'secondary_link': 40,
                    'tertiary': 40,
                    'tertiary_link': 40}

        # Plug in to the graph. 'speed_kph' attribute is added to each edge and set to the default speed.
        G = ox.add_edge_speeds(G, hwy_speeds)

        # Add 'travel_time' property to each node, which is 'length' divided by 'speed_kph'
        G = ox.add_edge_travel_times(G)
        travel_time = nx.get_edge_attributes(G, 'travel_time')
        nx.set_edge_attributes(G, travel_time,'total_delay') # initialize 'total_delay' attributes
        nx.set_edge_attributes(G, travel_time,'expected_delay') # initialize 'expected_delay' attributes

        self.G = G

    def init_env(self):
        self.env = simpy.Environment()
        self.tg = Traffic_Gen(self.env, self.G) 
        self.mv = Moving_Process(self.env, self.G, self.tg)
        self.edge_atk = Edge_Attack(self.env, self.G, self.tg) 

        self.rr = []
        if GV.REROUTE == True:
            for i in range(0, GV.ROUTE_GRP_NUM):
                grp_num = i
                intv = GV.REROUTE_INTV[i]
                self.rr.append(Reroute_Process(self.env, self.G, self.tg, grp_num, intv))
    
    def get_state(self):
        edge_cap = {}
        for edge in self.G.edges:
            edge_cap[edge] = self.G.edges[edge]['length'] * self.G.edges[edge]['lanes']
        edge_sat = nx.get_edge_attributes(self.G, 'saturation')
        visit_cnt = nx.get_edge_attributes(self.G, 'edge_cnt')

        nx.set_edge_attributes(self.G, 0, '1hop_cnt')
        nx.set_edge_attributes(self.G, 0, '2hop_cnt')
        for e in self.G.edges:
            for v in self.tg.Q_dic[e]:
                current_node_idx = v.e_idx
                next_node_idx = v.e_idx + 1
                sec_next_node_idx = v.e_idx +2

                if next_node_idx < len(v.path):
                    current_node = v.path[current_node_idx]
                    next_node = v.path[next_node_idx]
                    next_edge = (current_node, next_node, 0)
                    self.G.edges[next_edge]['1hop_cnt'] += 1
                
                if sec_next_node_idx < len(v.path):
                    sec_next_node = v.path[sec_next_node_idx]
                    sec_next_edge = (next_node, sec_next_node, 0)
                    self.G.edges[sec_next_edge]['2hop_cnt'] += 1
        
        one_hop_cnt = nx.get_edge_attributes(self.G, '1hop_cnt')
        two_hop_cnt = nx.get_edge_attributes(self.G, '2hop_cnt')
        is_alive = nx.get_edge_attributes(self.G, 'alive')
        
        # Get betweenness centrality of each edge
        D = nx.DiGraph(self.G)
        raw_bc = nx.edge_betweenness_centrality(D, weight='total_delay')

        # convert (u,v) format to (u,v,k) format
        edge_bc = {}
        for (u, v), value in raw_bc.items():
            edge_bc[(u, v, 0)] = value
        
        # Get the k-candidate edges
        cand_dic = {}
        cnt = nx.get_edge_attributes(self.G, 'edge_cnt')

        removed_edges = list(self.edge_atk.hist.values())
        for e in removed_edges:
            del cnt[e]

        sorted_cnt = sorted(cnt.items(), reverse=True, key=lambda d:d[1])

        for i, d in enumerate(sorted_cnt[:self.NUM_CAND]):
            edge = d[0]
            rank = i
            cand_dic[edge] = rank
        
        self.edge_atk.cand = cand_dic

        
        # Get status of edges and vectorize it
        edge_vecs = []

        for edge in self.G.edges:
            edge_stat = np.array([edge_cap[edge], edge_sat[edge], visit_cnt[edge], one_hop_cnt[edge], two_hop_cnt[edge], is_alive[edge], edge_bc[edge]])
            if edge in cand_dic:
                cand_rank = cand_dic[edge]
                cand_vec = self.ONEHOT_CAND[cand_rank]
            else:
                cand_vec = np.full(self.NUM_CAND, 0)

            edge_vec = np.append(edge_stat, cand_vec)
            edge_vecs.append(edge_vec)
        
        # Concatenate state information
        ob = np.concatenate(edge_vecs)

        return ob
        
    def disrupt(self, action):
        self.edge_atk.disrupt_order = action
        elapsed_time = GV.WARMING_UP + self.edge_atk.atk_rate * (self.edge_atk.atk_cnt + 1)
        self.env.run(until=elapsed_time)
        


