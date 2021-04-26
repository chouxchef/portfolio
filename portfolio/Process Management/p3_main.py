#
# CSCI 4210
# Project 3
# Jon Chernysh (chernj2)
# Alex Arsenault (arsena3)
#

from __future__ import division
import sys
import re
from operator import itemgetter, attrgetter
import copy

# take all process data from input file and place in list, return list of lists
def siphon(file):
	ps = []
	for line in file:
		if line.startswith('#') == False and line.startswith('\n') == False:
			line = re.sub('[\s]*', '', line)
			temp = [str(x) for x in line.split('|')]
			'''
			for x in temp:
				try:
					x = int(x)
				except:
					pass
				#print type(x), x
				'''
			ps.append(temp)
	return ps

class Process:
	def __init__(self, attribs, results):
		self.l = {}
		for index in range(len(results)):
			self.l[attribs[index]] = results[index]
	def __eq__(self, other):
		return self.l['proc-num'] == other.l['proc-num']
	def use(self, atr):
		try:
			return self.l[atr]
		except:
			return "-1.5"

class Obj:
	def __init__(self, attribs, results):
		self.v = []
		self.l = {}
		for index in range(len(results)):
			self.l[attribs[index]] = results[index]
	def use(self, atr):
		try:
			return self.l[atr]
		except:
			return "exception"

class Memory:
	def __init__(self, algo):
		self.alg = algo
		self.free_space = [[0, 255]] # list to track where blocks of space are located
		self.proc_space = [] # list to track where processes are located
		self.layout = ['.'] * 256 # visual layout of memory space, full of letters/dots
		self.last_index = 0 # most recent index referenced when process added
	
	def clear(self):
		self.free_space = [[0, 255]]
		self.proc_space = []
		self.layout = ['.'] * 256
		self.last_index = 0
	
	# sort the free_space and proc_space lists in increasing order of start_index
	def sortMemoryLists(self):
		self.free_space.sort(key=lambda tup: tup[0])
		self.proc_space.sort(key=lambda tup: tup[0])
		
	# print memory space contained in self.layout
	def printMemory(self, input=None):
		if input is None:
			input = self.layout
		line_length = 32
		print '=' * line_length
		for i in range(0, len(input), line_length):
			print ''.join(map(str, input[i:i+line_length]))
		print '=' * line_length
	
	
	# This function will add a process to memory according to any of the three placement
	# algorithms: first-fit, next-fit, and best-fit.
	def addProcess(self, proc):
		sufficient_memory = True # set to false if process cannot be added
		blocks_moved = 0 # number of blocks moved during defragmentation
		defragged = copy.deepcopy(self.layout)
		post = None
		# first need to be sure lists are in proper order
		self.sortMemoryLists()
		# First-fit: add process to the first block of memory that can fit it
		if self.alg == "first":
			# keep track of whether process was successfully added to memory
			proc_added = False
			for block in self.free_space:
				# if a large enough block of space is found, then update lists
				if 1+block[1]-block[0] >= proc.l['memory']:
					# add tuple of (block_start, block_end, proc_num) to proc_space
					self.proc_space.append([block[0], block[0]+proc.l['memory']-1, proc.l['proc-num']])
					# update displayed memory space to reflect new process
					for i in range(block[0], block[0]+proc.l['memory']):
						self.layout[i] = proc.l['proc-num']
					# update start index of free space block
					block[0] += proc.l['memory']
					proc_added = True
					break
			# if process not added successfully, try defragmenting
			if not proc_added:
				blocks_moved, post = self.defragmentMemory()
				# if defragmentation cleared up enough space, then try adding again
				if self.free_space[0][1]-self.free_space[0][0] >= proc.l['memory']:
					(sufficient_memory, temp_blocks_moved, tmp, post) = self.addProcess(proc)
				else:
					sufficient_memory = False
					
		# Next-fit: add process to the first memory block after previous added process
		# TODO: actually write this
		if self.alg == "next":
			# track whether or not the algorithm has reached bottom previously
			looped = False # set to true once process is added or once looped back to index
			proc_added = False
			second_pass = False # second pass on array, when looping around from bottom
			temp_index = self.last_index
			while not looped:
				for block in self.free_space:
					# if the search for a free block wrapped around, then search until
					# the location of the previous index
					if second_pass and block[1] >= self.last_index:
						looped = True
						break
					# if a large enough block of space is found after last_index then add
					# otherwise, loop back to top and search until last_index
					if 1+block[1]-block[0] >= proc.l['memory'] and temp_index <= block[1]:
						# add tuple of (block_start, block_end, proc_num) to proc_space
						self.proc_space.append([block[0], block[0]+proc.l['memory']-1, proc.l['proc-num']])
						# update displayed memory space to reflect new process
						for i in range(block[0], block[0]+proc.l['memory']):
							self.layout[i] = proc.l['proc-num']
						# update start index of free space block
						block[0] += proc.l['memory']
						self.last_index = block[1]
						proc_added = True
						break
						
				if proc_added:
					looped = True
					break
				else:
					second_pass = True
					temp_index = 0
			# if process not added successfully, try defragmenting
			if not proc_added:
				blocks_moved, post = self.defragmentMemory()
				# if defragmentation cleared up enough space, then try adding again
				if self.free_space[0][1]-self.free_space[0][0] >= proc.l['memory']:
					(sufficient_memory, temp_blocks_moved, tmp, post) = self.addProcess(proc)
				else:
					sufficient_memory = False
				
		# Best-fit: find the smallest free block that will fit the process
		# TODO: also write this
		if self.alg == "best":
			# keep track of whether process was successfully added to memory
			proc_added = False
			best_end_index = 255
			best_block_size = 256
			
			# find smallest free space block that can fit process
			for block in self.free_space:
				if 1+block[1]-block[0] < best_block_size and 1+block[1]-block[0] >= proc.l['memory']:
					best_end_index = block[1]
					best_block_size = 1+block[1]-block[0]
					
			for block in self.free_space:
				# if a large enough block of space is found, then update lists
				if block[1] == best_end_index and 1+block[1]-block[0] >= proc.l['memory']:
					# add tuple of (block_start, block_end, proc_num) to proc_space
					self.proc_space.append([block[0], block[0]+proc.l['memory']-1, proc.l['proc-num']])
					# update displayed memory space to reflect new process
					#print block,proc.l['memory']
					if block[0]+proc.l['memory'] > len(self.layout):
						proc_added = False
						break
					for i in range(block[0], block[0]+proc.l['memory']):
						#print i
						try:
							self.layout[i] = proc.l['proc-num']
						except:
							print "exception in memory allocation"
					# update start index of free space block
					block[0] += proc.l['memory']
					proc_added = True
					break
			# if process not added successfully, try defragmenting
			if not proc_added:
				blocks_moved, post = self.defragmentMemory()
				# if defragmentation cleared up enough space, then try adding again
				if self.free_space[0][1]-self.free_space[0][0] >= proc.l['memory']:
					(sufficient_memory, temp_blocks_moved, tmp, post) = self.addProcess(proc)
				else:
					sufficient_memory = False
					
		return (sufficient_memory, blocks_moved, defragged, post)

	# This function will delete a process from memory and update the memory layout
	# and lists appropriately. This occurs only once a process has completed its
	# CPU bursts. TODO: make use of sorting (sort by start index)
	def removeProcess(self, proc):
		# find the process in proc_space
		temp_proc_space = copy.deepcopy(self.proc_space)
		for block in temp_proc_space:
			if block[2] == proc.l['proc-num']:
				# update the displayed memory layout
				for c in range(len(self.layout)):
					if self.layout[c] == proc.l['proc-num']:
						self.layout[c] = '.'
				# update the free space list
				for index, free_block in enumerate(self.free_space):
					# if there is a free block immediately before this process
					# then change the end index of the block to that of the process
					if free_block[1] == block[0] - 1:
						free_block[1] = block[1]
						# see if there is another free block after the process
						if index < len(self.free_space)-1:
							# update end index and delete free block after process
							# NOTE: requires list to be sorted by start index
							if self.free_space[index+1][0] == free_block[1]+1:
								free_block[1] = self.free_space[index+1][1]
								del self.free_space[index+1]
								break
					# if there is a free block immediately after this process
					# then change the start index of the block to that of the process
					if free_block[0] == block[1] + 1:
						free_block[0] = block[0]
				# delete the process from proc_space
				self.proc_space = [[start, end, name] for start, end, name in self.proc_space 
										if name != block[2]]
			
	# This function defragments the memory space, moving processes closer to the top if
	# possible in order to consolidate free blocks in a single large block.
	# Returns the number of blocks moved (for time calculation purposes)
	def defragmentMemory(self):
		blocks_moved = 0 # total blocks moved during defragmentation
		total_free_blocks = 0 # total blocks free in memory
		prev_end_index = 0 # end index of process before one being considered
		
		# determine total amount of free memory
		for block in self.free_space:
			total_free_blocks += 1+block[1]-block[0]
		
		# move each process in memory to a position directly following another
		for block in self.proc_space:
			process_size = 1+block[1]-block[0]
			if block[0] != 0:
				if prev_end_index != block[0]-1:
					block[0] = prev_end_index + 1
					block[1] = block[0] + process_size - 1
					#print block
				for i in range(block[0], block[1]):
					self.layout[i] = block[2]
					blocks_moved += 1
			prev_end_index = block[1]
		
		# update free space list as well as visual representation
		self.free_space = [[256-total_free_blocks, 255]]
		for i in range(256-total_free_blocks, 255):
			self.layout[i] = '.'
			
		return blocks_moved, copy.deepcopy(self.layout)

class SimOut:
	# instance variables
	def __init__(self, fille):
		self.alg = ""
		self.c_s_counter = 0
		self.waits = []
		self.turnarounds = []
		self.bursts = []
		self.total = 0
		self.defrag = 0
		self.f = fille
	
	# print algorithm stats to output file
	def printAll(self):
		with open(self.f, "a") as f:
			f.write("Algorithm " + self.alg + "\n")
			f.write("-- average CPU burst time: " + str(round(self.average("bursts"), 2)) + " ms\n")
			f.write("-- average wait time: " + str(round(self.average("waits"), 2)) + " ms\n")
			f.write("-- average turnaround time: " + str(round(self.average("turnarounds"), 2)) + " ms\n")
			f.write("-- total number of context switches: " + str(self.c_s_counter) + "\n")
			f.write("-- time spent performing defragmentation: " + str(self.defrag) + "\n")
			f.write("-- percent spent performing defrag: " + str(round((self.defrag / self.total), 4)) + "\n")
	
	# given type of stat, add recorded time for each occurrence and find average
	def average(self, which):
		addup = float(0)
		if which == "waits":
			for i in self.waits:
				addup += i
			try:
				addup /= len(self.waits)
			except:
				pass
		if which == "turnarounds":
			for i in self.turnarounds:
				addup += i
			try:
				addup /= len(self.turnarounds)
			except:
				pass
		if which == "bursts":
			for i in self.bursts:
				addup += i
			addup /= len(self.bursts)
		return addup
	
	# re-initialize instance vars
	def clear(self):
		self.alg = ""
		self.c_s_counter = 0
		self.waits = []
		self.turnarounds = []
		self.bursts = []
		self.defrag = 0
		self.total = 0

class OpSys:
	def __init__(self, numCPU, algo):
		self.Q = [] # process queue
		self.IO = [] # IO queue
		self.num = numCPU # number of CPUs (for threading?)
		self.CPUs = [Obj(['context', 'ready', 'empty'], [0, 'no', 'yes'])]# array of CPUs
		self.alg = algo
		self.time = 0

	def printEvent(self, input, q_print = None):
		print "time " + str(self.time) + "ms: " + input,
		if q_print is None:
			if len(self.Q) == 0:
				print "[Q]"
			else:
				print "[Q",
				for i in range(len(self.Q)):
					if i < len(self.Q)-1:
						print self.Q[i].l['proc-num'],
					else:
						print str(self.Q[i].l['proc-num']) + "]"
		else:
			print
	
	def checkCPU(self, num):
		res = [0, 0]
		if self.CPUs[num].l['empty'] == 'no':
			self.CPUs[num].l['ready'] = 'no'
			if self.CPUs[num].v[0].l['finish-at'] == self.time:
				if self.CPUs[num].v[0].l['num-burst'] > 1:
					self.printEvent("Process \'" + str(self.CPUs[num].v[0].l['proc-num']) + "\' completed its CPU burst")
					self.CPUs[num].v[0].l['finish-at'] = self.time + self.CPUs[num].v[0].l['io-time']
					sim.turnarounds.append(self.CPUs[num].v[0].l['turnaround'])
					sim.waits.append(self.CPUs[num].v[0].l['wait'])
					self.IO.append(self.CPUs[num].v[0])
					self.printEvent("Process \'" + str(self.IO[-1].l['proc-num']) + "\' performing I/O")
				else:
					res = self.CPUs[num].v[0], 'terminated'
				self.CPUs[num].l['empty'] = 'yes'
				self.CPUs[num].l['ready'] = 'maybe'
		if self.CPUs[num].l['empty'] == 'yes':
			if self.CPUs[num].l['context'] == self.time:
				self.CPUs[num].l['ready'] = 'yes'
		if res[1] == 'terminated':
			return res
		else:
			return 0, 'nada'

	def checkIO(self):
		temp = []
		if len(self.IO) != 0:
			for bundle in self.IO:
				if self.time >= bundle.l['finish-at']:
					temp.append(bundle)
		self.IO[:] = (x for x in self.IO if x.l['finish-at'] != self.time)
		for t in temp:
			t.l['num-burst'] -= 1
			t.l['remaining-time'] = t.l['burst-time']
			t.l['wait'] = 0
			t.l['turnaround'] = 0
		return temp
    
	def q_addition(self, to_add, cpu_num):
		temp = []
		for t in to_add:
			if t.l['arrival-time'] > self.time:
				temp.append(t)
		for t in temp:
			to_add.remove(t)
		if self.alg == "SRT":
			for element in to_add:
				if self.CPUs[cpu_num].l['empty'] == 'no':
					if self.CPUs[cpu_num].v[0].l['remaining-time'] > element.l['burst-time']:
						return element
				self.Q.append(element)
			#self.Q = sorted(self.Q, key=attrgetter('remaining-time', 'proc-num'))
			self.Q = sorted(self.Q, key=lambda x: x.l['remaining-time'])
		if self.alg == "RR":
			if self.time == 0:
				self.Q.extend(to_add)
			else:
				self.Q.extend(sorted(to_add, key=lambda x: x.l['proc-num']))
		return 0
	
	def CPAdd(self, switch):
		switch.l['finish-at'] = self.time + switch.l['remaining-time']
		sim.c_s_counter += 1
		try:
			self.CPUs[0].v[0] = switch
		except:
			self.CPUs[0].v.append(switch)
		self.CPUs[0].l['empty'] = 'no'
		self.CPUs[0].v[0].l['elapsed'] = 0
	
	def memPrint(self, t_slice, input, blocks):
		if input[4] is not None:
			self.printEvent("Process \'" + str(input[0].l['proc-num']) + "\' unable to be added; lack of memory", False)
			self.printEvent("Starting defragmentation (suspending all processes)", False)
			self.printEvent("Simulated Memory:", False)
			blocks.printMemory(input[3])
		self.time += input[2] * 10
		try:
			self.CPUs[0].v[0].l['finish-at'] += input[2] * 10
		except:
			pass
		for i in range(len(self.Q)):
			self.Q[i].l['wait'] += input[2] * 10
			self.Q[i].l['turnaround'] += input[2] * 10
		if self.CPUs[0].l['empty'] == 'no':
			self.CPUs[0].v[0].l['turnaround'] += input[2] * 10
		if input[4] is not None:
			self.printEvent("Completed defragmentation (moved " + str(input[2]) + " memory units)", False)
			self.printEvent("Simulated Memory:", False)
			blocks.printMemory(input[4])
		if input[1]:
			self.printEvent("Process \'" + str(input[0].l['proc-num']) + "\' added to system")
			self.printEvent("Simulated Memory:", False)
			blocks.printMemory()
		else:
			self.printEvent("Process \'" + str(input[0].l['proc-num']) + "\' could not be added, ignoring request", False)
		return input[2] * 10
	
	def simulate(self, procs, n, t_cs, t_slice, sim, memalg, attribs):
		self.time = 0
		sim.clear()
		blocks = Memory(memalg)
		blocks.clear()
		if self.alg == "RR":
			inser = self.alg + " (t_slice " + str(t_slice) + ")"
		else:
			inser = self.alg
		self.printEvent("Simulator started for " + inser + " and " + str(memalg) + "-fit", False)
		sim.alg = self.alg
		upload = copy.deepcopy(procs)
		for u in upload:
			for e in u:
				try:
					u[u.index(e)] = int(e)
				except:
					continue
		processes = []
		for p in upload:
			for i in range(p[attribs.index('num-burst')]):
				sim.bursts.append(p[attribs.index('burst-time')])
			processes.append(Process(attribs, p))
			processes[-1].l['finish-at'] = -10
			processes[-1].l['remaining-time'] = processes[-1].l['burst-time']
			processes[-1].l['wait'] = 0
			processes[-1].l['turnaround'] = 0
			if processes[-1].use('arrival-time') == '-1.5':
				processes[-1].l['arrival-time'] = 0
		self.p_size = len(upload[0])
		self.CPUs[0].l['empty'] = 'yes'
		to_add = copy.deepcopy(processes)
		for p in processes:
			if p.l['arrival-time'] == self.time:
				temp = [p]
				temp.extend(blocks.addProcess(p))
				sim.defrag += self.memPrint(t_slice, temp, blocks)
				if temp[1]:
					self.q_addition([p], 0)
				to_add.remove(p)
		self.CPUs[0].l['context'] = t_cs
		self.CPUs[0].l['ready'] = 'maybe'
		while True:
			checker = self.checkCPU(0)
			if checker[1] == 'terminated':
				blocks.removeProcess(checker[0])
				self.printEvent("Process \'" + str(checker[0].l['proc-num']) + "\' terminated")
				self.printEvent("Simulated Memory:", False)
				blocks.printMemory()
			for elem in to_add:
				if elem.l['arrival-time'] == self.time:
					temp = [elem]
					temp.extend(blocks.addProcess(elem))
					if temp[1]:
						self.q_addition([elem], 0)
					sim.defrag += self.memPrint(t_slice, temp, blocks)
			for p in self.Q:
				if p in to_add:
					to_add.remove(p)
			if self.CPUs[0].l['ready'] == 'yes':
				tmp = 0
				if self.CPUs[0].use('preempt') != 'exception':
					tmp = self.CPUs[0].l.pop('preempt', None)
				elif len(self.Q) > 0:
					tmp = self.Q.pop(0)
				if tmp != 0:
					self.CPAdd(tmp)
					self.printEvent("Process \'" + str(self.CPUs[0].v[0].l['proc-num']) + "\' started using the CPU")
			if self.CPUs[0].l['empty'] == 'yes':
				if len(self.Q) == 0 and len(self.IO) == 0:
					break
			else:
				if self.alg == 'RR':
					if self.CPUs[0].v[0].l['elapsed'] == t_slice:
						if len(self.Q) > 0:
							self.q_addition([self.CPUs[0].v[0]], 0)
							self.printEvent("Process \'" + str(self.Q[-1].l['proc-num']) + "\' preempted due to time slice expiration")
							self.CPUs[0].l['empty'] = 'yes'
							self.CPUs[0].l['ready'] = 'maybe'
						else:
							self.CPUs[0].v[0].l['elapsed'] = 0
			slated = copy.deepcopy(self.checkIO())
			results = self.q_addition(slated, 0)
			if self.CPUs[0].l['empty'] == 'yes' and (len(self.Q) > 0 or len(slated) > 0) and self.CPUs[0].l['ready'] == 'maybe':
				self.CPUs[0].l['context'] = self.time + t_cs
				self.CPUs[0].l['ready'] = 'no'
			for s in slated:
				self.printEvent("Process \'" + str(s.l['proc-num']) + "\' completed I/O")
			if results != 0:
				tmp = self.CPUs[0].v[0]
				self.CPUs[0].l['empty'] = 'yes'
				self.q_addition([tmp], 0)
				self.printEvent("Process \'" + str(tmp.l['proc-num']) + "\' preempted by process \'" + str(results.l['proc-num']) + "\'")
				self.CPUs[0].l['context'] = self.time + t_cs
				self.CPUs[0].l['ready'] = 'no'
				self.CPUs[0].l['preempt'] = copy.deepcopy(results)
			
			self.time += 1
			if self.CPUs[0].l['ready'] =='no':
				if len(self.Q) > 0:
					self.Q[0].l['wait'] += 1
			for i in range(len(self.Q)):
				self.Q[i].l['turnaround'] += 1
				if i is not 0:
					self.Q[i].l['wait'] += 1
			try:
				self.CPUs[0].v[0].l['turnaround'] += 1
			except:
				pass
			if self.CPUs[0].l['empty'] == 'no':
				self.CPUs[0].v[0].l['remaining-time'] -= 1
				self.CPUs[0].v[0].l['elapsed'] += 1
		print "time " + str(self.time) + "ms: Simulator for " + str(self.alg) + " ended [Q]"
		sim.total = self.time
		sim.printAll()

if __name__ == "__main__":
	if len(sys.argv) > 2:
		print "Error: expected at most two command-line arguments:"
		print "1. The program file itself"
		print "2. The optional text file to input processes"
		sys.exit()
	if len(sys.argv) == 2:
		try:
			text = open(sys.argv[1])
		except IOError:
			print "Error: No such file exists in current directory."
			sys.exit()
	else:
		try:
			text = open("processes.txt")
		except IOError:
			print "Error: could not find \"processes.txt\" in current directory"
			sys.exit()
	procs = siphon(text)
	text.close()
	atrs = ["proc-num", "arrival-time", "burst-time", "num-burst", "io-time", "memory"]
	memalg = ("first", "next", "best")
	ops = OpSys(1, "SRT")
	sim = SimOut("simout.txt")
	fi = open("simout.txt", "w+")
	fi.close()
	for m in memalg:
		ops.simulate(procs, len(procs), 13, 80, sim, m, atrs)
		print
		print
	ops.alg = "RR"
	for m in memalg:
		ops.simulate(procs, len(procs), 13, 80, sim, m, atrs)
		print
		print