import networkx as nx
import osmnx as ox
import numpy as np
import random
import simpy
import pickle
import pkg_resources
import matplotlib.pyplot as plt
import os
import time

class GV:
    SIM_TIME = 15000 + 1
    GEN_RATE = 1.5
    GEN_END = 2000 * GEN_RATE
    ATK_RATE = 200

    MOVE_INTV = 1
    REROUTE_INTV = None
    VEHICLE_LENGTH = 4.5
    WARMING_UP = 1000

    LOG_DIR = os.getcwd() + '/results'
    ITER_NUM = 0
    START_WALL_TIME = None
    END_WALL_TIME = None

    BETA_1 = 1
    BETA_2 = 3
    BETA_3 = 20
    CP_1 = 0.5
    CP_2 = 0.8
    NEXT_CP = 0.5
    
    DISRUPT_TYPE = [
        'motorway', 'primary',
        'secondary', 'tertiary'
    ]

# Traffic generation process
class Traffic_Gen(object): 
    def __init__(self, env, G):
        self.env = env
        self.gen_rate = GV.GEN_RATE
        self.vehicle_number = 0
        self.G = G
        self.Q_dic = {}   # Dictionary of traffic queue for each edge in the network 
        self.delay_dic = {}   # Dictionary of practical delay log on each edge
        
        for edge in self.G.edges:
            self.Q_dic[edge] = []
            self.delay_dic[edge] = []
        
        # traffic demand group
        
        # UCD is selected as dst. at 50% prob.
        self.ucd = [
            2607436562,   # Southwest
            1910520967,   # South
            1910521041,   # Southeast
            2599137941, # North
            2607436619 # West
        ]
        
        # groceries are selected as dst. at 10% prob.
        self.groceries = [
            599394518, #Safeway (N) 
            95710949, #Safeway (S)
            3076147004, #Savemart
            95714491, #TJ
            2580278625, #Nugget (N)
            599381072, #Nugget (S)
            95714168, #Target
        ]

        # downtown area is selected as dst. at 10% prob.
        self.downtown = [
            95711354,
            95711358,
            95711361,
            95713482,
            271743555,
            498524153,
            498524168
        ]

        # I-80W (eastbound) is selected as org. at 15% prob.
        self.i80w = [
            267383938
        ]

        # I-80E (westbound) is selected as org. at 10% prob.
        self.i80e = [
            62224641
        ]
        
        self.nodes = list(self.G.nodes)
        
        # compute org. node sampling weight
        nx.set_node_attributes(self.G, 1, 'org_w')
        general_org_num = len(self.nodes) - len(self.i80w) - len(self.i80e)
        
        # Other nodes are sampled at 75% of probability
        i80w_prob = general_org_num * 15 / 75
        i80e_prob = general_org_num * 10 / 75

        for node in self.G.nodes():
            if node in self.i80w:
                self.G.nodes[node]['org_w'] = i80w_prob / len(self.i80w)
            elif node in self.i80e:
                self.G.nodes[node]['org_w'] = i80e_prob / len(self.i80e)
                
        self.org_weights = [self.G.nodes[node]['org_w'] for node in self.nodes]
        
        # compute dst. node sampling weight
        nx.set_node_attributes(self.G, 1, 'dst_w')
        general_dst_num = len(self.nodes) - len(self.ucd) - len(self.groceries) - len(self.downtown)

        # Other nodes are sampled at 30% of probability
        ucd_prob = general_dst_num * 5 / 3      # 50% from UCD
        downtown_prob = general_dst_num / 3     # 10% from downtown area
        groceries_prob = general_dst_num / 3    # 10% from groceries

        for node in self.G.nodes():
            if node in self.ucd:
                self.G.nodes[node]['dst_w'] = ucd_prob / len(self.ucd)
            elif node in self.downtown:
                self.G.nodes[node]['dst_w'] = downtown_prob / len(self.downtown)
            elif node in self.groceries:
                self.G.nodes[node]['dst_w'] = groceries_prob / len(self.groceries)
        
        self.dst_weights = [self.G.nodes[node]['dst_w'] for node in self.nodes]
        
        self.action = env.process(self.run())
        
    def run(self):       
        while True:
            # Infinite loop for generating traffic
            yield self.env.timeout(random.expovariate(self.gen_rate))
                           
            if self.vehicle_number >= GV.GEN_END : continue
            
            # Create and enqueue a new vehicle            
            generated = False
            
            while not generated:
                try :
                    # Generate a vehicle with random src/dst pair
                    src = random.choices(self.nodes, weights=self.org_weights, k=1)[0]
                    dst = random.choices(self.nodes, weights=self.dst_weights, k=1)[0]
                    
                    if src == dst:
                        raise ValueError('src and dst node are the same')
                        
                    path = nx.shortest_path(self.G, src, dst, weight='expected_delay')
                    start_edge = (path[0], path[1], 0)

                    new_vehicle = Vehicle(self.vehicle_number, self.env.now, src, dst, path)

                    # Put the vehicle in the starting edge
                    self.vehicle_entry(start_edge, new_vehicle)
                    self.G.edges[start_edge]['edge_cnt'] += 1
                    
                    self.vehicle_number += 1
                    generated = True

                except Exception as error:
                    # Errors above and there are some pairs that have no path between src/dst
                    #print(error)
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

        if edge_type == 'primary':
            signal_delay = 10

        elif edge_type == 'secondary':
            signal_delay = 10

        elif edge_type == 'tertiary':
            signal_delay = 6
                
        elif edge_type == 'residential':
            signal_delay = 4

        # Get penalty rate    

        # Is any of possible next edges congested?
        next_edge_congested = False
        (current_node, next_node, key) = edge

        for (u, v) in self.G.out_edges(next_node):
            if v != current_node:
                if self.G.edges[(u, v, 0)]['saturation'] > GV.NEXT_CP:
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
    def __init__(self, identifier, gen_time, src, dst, path):
        # Basic stats
        self.identifier = identifier
        self.gen_time = gen_time
        self.src = src
        self.dst = dst
        
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
        
        self.v_num = [0]

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
                            
            epsilon = 0.01
            
            log_time = self.env.now % 20
            #print_time = self.env.now % 100
            
            if log_time < epsilon or abs(log_time) > 20 - epsilon: 
                # Log the number of vehicles in the network in every 20 seconds
                vn = 0
                for queue in self.tg.Q_dic.values():
                    vn += len(queue)
                
                self.v_num.append(vn)
                
            #if print_time < epsilon or abs(print_time) > 100 - epsilon: print("Simulation time: {0}".format(self.env.now))
            
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

        if edge_type == 'primary':
            signal_delay = 10

        elif edge_type == 'secondary':
            signal_delay = 10

        elif edge_type == 'tertiary':
            signal_delay = 6
                
        elif edge_type == 'residential':
            signal_delay = 4

        # Get penalty rate    

        # Is any of possible next edges congested?
        next_edge_congested = False
        (current_node, next_node, key) = edge

        for (u, v) in self.G.out_edges(next_node):
            if v != current_node:
                if self.G.edges[(u, v, 0)]['saturation'] > GV.NEXT_CP:
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
    def __init__(self, env, G, traffic_generator):
        self.env = env
        self.interval = GV.REROUTE_INTV
        self.G = G   
        self.tg = traffic_generator
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.interval)

            for edge in self.G.edges:
                q = self.tg.Q_dic[edge]
                for vehicle in q:
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
        self.max_cnt = 5

        # interprete action to target edge
        self.target = None
        self.candidates = None
        self.past_actions = []
        self.last_atk_time = None

        # Term edge count after idx-th attack
        self.term_edge_cnt = []

        self.action = env.process(self.run())
        
    def run(self):       
        yield self.env.timeout(GV.WARMING_UP) # defer attack to the end of warming-up period
        while True:                        
            if self.atk_cnt < self.max_cnt:
                # Select an edge to disrupt that chosen by the agent
                for idx, edge in enumerate(self.candidates):
                    if idx == self.target:
                        vul_edge = edge
                        self.past_actions.append(self.target)
                        self.target = None # initialize target (given action)
                
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
                self.reroute()

                self.atk_cnt += 1
                self.last_atk_time = self.env.now

            yield self.env.timeout(self.atk_rate)

    def reroute(self): 
        for edge in self.G.edges:
            q = self.tg.Q_dic[edge]
            for vehicle in q:
                # the index of the node that this vehicle is moving toward on the current edge
                next_node_idx = vehicle.e_idx + 1

                next_node = vehicle.path[next_node_idx]
                left_path = vehicle.path[next_node_idx:]
                new_path = nx.shortest_path(self.G, next_node, vehicle.dst, weight='expected_delay')

                if left_path != new_path: # reroute if there is a shorter path
                    history = vehicle.path[:next_node_idx]
                    new_route = history + new_path
                    vehicle.path = new_route

class ROADNET(object):
    def __init__(self):
        # Define variables
        self.G = None
        self.env = None
        self.tg = None
        self.mv = None
        self.edge_atk = None
        self.rr = None

        self.init_graph()

    def init_graph(self):
        # load road network from a binary pickle file 
        self.G = pickle.load(pkg_resources.resource_stream(__name__, 'data/Davis_super_simplified_graph.pkl'))

        candidates = []
        for u,v,k,d in self.G.edges(keys=True, data=True):
            edge = (u,v,k)
            if d['highway'] in GV.DISRUPT_TYPE:
                candidates.append(edge)

        # initialize environment
        self.env = simpy.Environment()
        self.tg = Traffic_Gen(self.env, self.G) 
        self.mv = Moving_Process(self.env, self.G, self.tg)
        self.edge_atk = Edge_Attack(self.env, self.G, self.tg) 
        self.edge_atk.candidates = candidates

        GV.START_WALL_TIME = time.time()

        self.env.run(GV.WARMING_UP) # run right before the first disruption
    
    def get_state(self):
        Q_len = dict.fromkeys(self.G.edges, 0)
        fwd_cnt = dict.fromkeys(self.G.edges, 0)

        for edge in self.G.edges:
            Q = self.tg.Q_dic[edge]
            Q_len[edge] = len(Q)
            for vehicle in Q:
                current_node_idx = vehicle.e_idx
                next_node_idx = vehicle.e_idx + 1
                current_node = vehicle.path[current_node_idx]
                last_src_node = vehicle.path[-2]
                if current_node != last_src_node:
                    start_nodes = vehicle.path[current_node_idx:-2]
                    end_nodes = vehicle.path[next_node_idx:-1]
                    keys = [0] * len(start_nodes)  # we use simplified graph without multi-edges
                    
                    forward_edges = zip(start_nodes, end_nodes, keys)
                    for fwd_edge in forward_edges:
                        fwd_cnt[fwd_edge] += 1
        
        # Get status of edges and vectorize it
        edge_vecs = []

        for u,v,k,d in self.G.edges(keys=True, data=True):
            edge_len = int(d['length'])
            edge_cap = int(edge_len * int(d['lanes']) / GV.VEHICLE_LENGTH)
            vis_cnt = int(d['edge_cnt'])
            is_alive = int(d['alive'])
            edge_Q_len = int(Q_len[(u,v,k)])
            edge_fwd_cnt = int(fwd_cnt[(u,v,k)])
            speed_limit = int(d['speed_kph'])

            edge_stat = np.array([edge_cap, edge_Q_len, vis_cnt, edge_fwd_cnt, is_alive, speed_limit, edge_len])

            edge_vecs.append(edge_stat)
        
        # Concatenate state information
        ob = np.concatenate(edge_vecs)

        return ob
        
    def disrupt(self, action):
        self.edge_atk.target = action
        is_dup_action = False
        
        if action in self.edge_atk.past_actions:
            is_dup_action = True

        if self.edge_atk.atk_cnt == 4:
            self.env.run(until=GV.SIM_TIME)
            GV.END_WALL_TIME = time.time()
            self.save_log()
        else:
            elapsed_time = GV.WARMING_UP + self.edge_atk.atk_rate * (self.edge_atk.atk_cnt + 1)
            self.env.run(until=elapsed_time)

        return is_dup_action
        

    def save_log(self):
        # Plot attacked edges on the map
        dir = GV.LOG_DIR + '/' + str(GV.ITER_NUM) + '/'
        os.makedirs(dir)

        hist_list = list(self.edge_atk.hist.values())

        removed = []
        for e in hist_list:
            removed.append( [e[0], e[1]] )

        c_pool = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        rc = []

        for i in range(0, len(removed)):
            rc.append(c_pool[i])

        atk_fig, atk_ax = ox.plot.plot_graph_routes(self.G, removed, figsize=(30,18), route_colors=rc, 
                                                    route_linewidth=6, route_alpha=0.8, orig_dest_size=100,
                                                    ax=None, node_size=20, node_color='black',
                                                    edge_linewidth=0.5, edge_color='black', 
                                                    show=False, close=False, bgcolor='white')

        for i, e in enumerate(removed):
            start_node = e[0]
            end_node = e[1]
            start_coord = (self.G.nodes[start_node]['x'] - 0.0006, self.G.nodes[start_node]['y'] - 0.0006)
            end_coord =  (self.G.nodes[end_node]['x'] - 0.0006, self.G.nodes[end_node]['y'] - 0.0006)

            # Annotate node ID if needed
            #atk_ax.annotate(start_node, xy=start_coord, color='white', weight='bold')
            #atk_ax.annotate(end_node, xy=end_coord, color='blue', weight='bold')

            middle_coord = (start_coord[0] + 0.5*(end_coord[0] - start_coord[0]), start_coord[1] + 0.5*(end_coord[1] - start_coord[1]))

            arrow_len = 0.001

            x_s = start_coord[0]
            y_s = start_coord[1]
            x_e = end_coord[0]
            y_e = end_coord[1]
            x_m = middle_coord[0]
            y_m = middle_coord[1]

            x_diff = x_s - x_m
            y_diff = y_s - y_m
            grad = y_diff / x_diff

            x_sign = x_diff / abs(x_diff)
            y_sign = y_diff / abs(y_diff)

            arrow_start_coord = (middle_coord[0] + 0.3*(start_coord[0] - middle_coord[0]), middle_coord[1] + 0.3*(start_coord[1] - middle_coord[1]))
            arrow_end_coord = (middle_coord[0] + 0.3*(end_coord[0] - middle_coord[0]), middle_coord[1] + 0.3*(end_coord[1] - middle_coord[1]))    
            dist = (((arrow_start_coord[0]-arrow_end_coord[0])**2 + (arrow_start_coord[1] - arrow_end_coord[1])**2)**0.5)

            if dist < 0.003:
                    x_mov = x_sign*((arrow_len**2 / (1 + grad**2))**0.5)
                    y_mov = y_sign*((arrow_len**2 - x_mov**2) ** 0.5)
                    arrow_start_coord = (middle_coord[0] + x_mov, middle_coord[1] + y_mov)
                    arrow_end_coord = (middle_coord[0] - x_mov, middle_coord[1] - y_mov)

            atk_ax.annotate('', xy= arrow_end_coord, color='blue', weight='bold', xytext= arrow_start_coord, textcoords='data', arrowprops=dict(color=rc[i], headwidth=10, headlength=10, lw=1))

        # Add legend
        line_label, = plt.plot([0, 0, 0], label='1st', c='none', linewidth = 0)
        line_1st, = plt.plot([0, 0, 0], label='1st', c=rc[0], linewidth = 6)
        line_2nd, = plt.plot([0, 0, 0], label='2nd', c=rc[1], linewidth = 6)
        line_3rd, = plt.plot([0, 0, 0], label='3rd', c=rc[2], linewidth = 6)
        line_4th, = plt.plot([0, 0, 0], label='4th', c=rc[3], linewidth = 6)
        line_5th, = plt.plot([0, 0, 0], label='5th', c=rc[4], linewidth = 6)


        plt.legend([line_label, line_1st, line_2nd, line_3rd, line_4th, line_5th], ['Top-5 edges', '1st', '2nd', '3rd', '4th', '5th'], prop={'size': 20})

        plt.savefig(dir + 'Attacked_edges_map.png')
        plt.close('all')
        
        disrupted = self.edge_atk.hist.values()

        disrupted_fname = dir + 'disrupted.txt'
        with open(disrupted_fname, "w") as output:
            for value in disrupted:
                output.write(str(value) + '\n')

        # Save travel time
        finished_travelTime = []
        for vehicle in self.mv.finished:
            travelTime = vehicle.arrival_time - vehicle.gen_time
            finished_travelTime.append(travelTime)

        sorted_finished_travelTime = sorted(finished_travelTime, reverse=True)
        fd_fname = dir + 'finished_travelTime.txt'
        with open(fd_fname, "w") as output:
            for value in sorted_finished_travelTime:
                output.write(str(value) + '\n')

        onTheWay_travelTime = []
        for edge in self.G.edges:
            for vehicle in self.tg.Q_dic[edge]:
                travelTime = GV.SIM_TIME - vehicle.gen_time
                onTheWay_travelTime.append(travelTime)
                
        sorted_onTheWay_travelTime = sorted(onTheWay_travelTime, reverse=True)
        otw_fname = dir + 'onTheWay_travelTime.txt'
        with open(otw_fname, "w") as output:
            for value in sorted_onTheWay_travelTime:
                output.write(str(value) + '\n')
                
        # Save general stats
        stat_fname = dir + 'stats.txt'

        finished_sum = sum(finished_travelTime)
        otw_sum = sum(onTheWay_travelTime)
        total_travelTime = finished_sum + otw_sum

        with open(stat_fname, "w") as output:
            output.write(str(self.tg.vehicle_number) + ' vehicles are generated' + '\n')
            output.write(str(len(self.mv.finished)) + ' vehicles completed travel' + '\n')
            output.write('Travel time sum. of finished travels: ' + str(finished_sum) + '\n')
            output.write('Travel time sum. of vehicles on their way: ' + str(otw_sum) + '\n')
            output.write('Total travel time sum.: ' + str(total_travelTime) + '\n')
            output.write('Simulation runtime: ' + str(GV.START_WALL_TIME - GV.END_WALL_TIME) + '\n')

        # Dump data for debugging
        dat_dir = dir + "var_dat/"
        os.mkdir(dat_dir)

        nx.write_gpickle(self.G, dat_dir + "graph.gpickle")

        with open(dat_dir + "Q_dic.p", 'wb') as f:
            pickle.dump(self.tg.Q_dic, f)
            
        with open(dat_dir + "delay_dic.p", 'wb') as f:
            pickle.dump(self.tg.delay_dic, f)
            
        with open(dat_dir + "finished.p", 'wb') as f:
            pickle.dump(self.mv.finished, f)

        with open(dat_dir + "v_num.p", 'wb') as f:
            pickle.dump(self.mv.v_num, f)


        vnum_fname = dir + 'vnum_gr-' + str(GV.GEN_RATE) + '_rr-' + str(GV.REROUTE_INTV) +'.txt'
        with open(vnum_fname, "w") as output:
            for value in self.mv.v_num:
                output.write(str(value) + '\n')

        num_dp = ((GV.SIM_TIME - 1) / 20) + 1
        x = [ i*20 for i in range(int(num_dp)) ]

        x2 = [ i*20 for i in range(int(num_dp))]
        y = [ t*GV.GEN_RATE for t in x2 ]

        #dsrp_x = [800, 1100, 1400, 1700, 2000]

        plt.plot(x, self.mv.v_num, label='Moving vehicles')
        plt.plot(x2, y, label='Generated')
                
        plt.title('The number of vehicles')
        plt.xlabel('Seconds') 
        plt.ylabel('Counts')

        plt.ylim(top=max(self.mv.v_num)*1.1, bottom=0)
        plt.xlim(left=0)
        plt.legend()

        plt.savefig(dir + 'num_vehicle.png')
        plt.close('all')

        GV.ITER_NUM += 1
        GV.START_WALL_TIME = None
        GV.END_WALL_TIME = None