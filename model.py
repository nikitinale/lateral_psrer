# :export code # для тангла кода
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main executable file of model for evaluation radioisotopes horisontal migration into close lake from its catchment area
"""

import sys
import os
import csv
import math
import numpy as np
from datetime import datetime
from tqdm import tqdm_notebook

np.random.seed(datetime.now().second)
datafile = 'perstok_data.csv'
outfile = 'perstok_new.csv'

cs137_decay_constant = 0.0229769
am241_decay_constant = 0.0016023
pu241_decay_constant = 0.0495
cs137_infiltration_constant = 0.0273
am241_infiltration_constant = 0.0298
cs137_infiltration_constant_2 = 0.0677
am241_infiltration_constant_2 = 0.0658

class SoilCell:
  """
  The class define cell with soil and plant cover wich is a main element of the model
  """
  def __init__(self, x, y, elevation, land_type, cs_137=30000.0, am_241=1000.0, cs_137b=3000.0, am_241b=100.0, size=100.0, soil_density=1.35, cs137_s_part=0.0015, w0=3.0):
    isotope_layer = {}
    self.x = x # position of a cell
    self.y = y # position of a cell
    self.soil_density = soil_density + 0.02*np.random.randn()
    self.elevation = elevation
    self.land_type = land_type # type of land cover: l - meadow, f - forest, w - covered with water
    self.input_cells = [] # cells below the current
    self.up_way_cells = [] # cells upward the current
    self.output_cell = None # cell which takes outflows from the current
    self.model_R = 600000.0 + 50*np.random.randn() # Parameters of Wischmeier and Smith's
                                                   # Empirical Soil Loss Model (USLE)
                                                   # denended of land cover type:
    self.model_P = 1
    if self.land_type == 'f':
      self.model_K = 0.020 + 0.001*np.random.randn()
      self.model_C = 0.006 + 0.0003*np.random.randn()
      self.model_LS = 0.1 + 0.001*np.random.randn()
    elif self.land_type == 'l':
      self.model_K = 0.0785 + 0.001*np.random.randn()
      self.model_C = 0.05 + 0.0003*np.random.randn()
      self.model_LS = 0.3 + 0.001*np.random.randn()
    else :
      self.model_K = 0.0
      self.model_C = 0.0
      self.model_LS = 0.0
    self.size = size # length of a side of a square cell, m
    self.angle = 0  # angle of inclination of the ground surface in a cell
    self.uklon = 0 # slope of the ground surface around a cell
    self.soil_loss = 0 # weight of soil lost from the cell with waterflows (kg/year)
    self.sediment_inflow = 0 # weight of soil brought into a cell with waterflows (kg/year)
    self.max_outflow_capicity = 0 # maximum possible weight of soil loss from a cell in one event
    self.w0 = w0 + 0.2*np.random.randn() # stack of free water in a migratory active layer of soil kg/sq.m.
    # properties of Cs-137 and Am-241 in top (0--2 cm) and bottom (2--20 cm) layers of soil
    self.cs137 = {'top_layer': {'activity_concentration': cs_137 + 0.05*cs_137*np.random.randn(), 
                           'zapas': 0.0, # total stack of the radioisotope
                           'soluble_part': cs137_s_part + 0.0002*np.random.randn(), 
                           'soluble_zapas': 0.0, # stock in soluble form
                           'activity_loss': 0.0, # loss of the radioisotope activity in absorbed on soil particles form
                           'liquid_flow': 0.0}, # loss of the radioisotope activity in dissolved  form
             'bottom_layer': {'activity_concentration': cs_137b + 0.05*cs_137b*np.random.randn(), 
                              'zapas': 0}, 
             'decay_constant': cs137_decay_constant,
             'infiltration_constant': cs137_infiltration_constant, # part of the activity transfered from top into bottom layer per year
             'inflow': 0.0, # activity of the radioisotope received by the cell per year
             'name': 'Cs-137'
             }
    self.cs137['top_layer']['zapas'] = self.size**2 * self.cs137['top_layer']['activity_concentration'] * self.soil_density*(0.2*10*10)
    self.cs137['bottom_layer']['zapas'] = self.size**2 * self.cs137['bottom_layer']['activity_concentration'] * self.soil_density*(1.8*10*10)
    self.cs137['top_layer']['soluble_zapas'] = self.cs137['top_layer']['soluble_part'] * self.cs137['top_layer']['zapas']

    self.am241 = {'top_layer': {'activity_concentration': am_241 + 0.1*am_241*np.random.randn(), 
                           'zapas': 0.0,
                           'activity_loss': 0.0,
                           'soluble_part': 0.0,
                           'soluble_zapas': 0.0,
                           'liquid_flow': 0.0}, 
             'bottom_layer': {'activity_concentration': am_241b + 0.05*am_241b*np.random.randn(), 
                              'zapas': 0},
             'decay_constant': am241_decay_constant,
             'infiltration_constant': am241_infiltration_constant,
             'inflow': 0.0,
             'name': 'Am-241'
             }
    self.am241['top_layer']['zapas'] = self.size**2 * self.am241['top_layer']['activity_concentration'] * self.soil_density*(0.2*10*10)
    self.am241['bottom_layer']['zapas'] = self.size**2 * self.am241['bottom_layer']['activity_concentration'] * self.soil_density*(1.8*10*10)
    if self.land_type == 'f':
      self.cs137['infiltration_constant'] = cs137_infiltration_constant_2
      self.am241['infiltration_constant'] = am241_infiltration_constant_2
    self.pu241 = {'top_layer': {'activity_concentration': 0.0, 
                           'zapas': 0.0,
                           'activity_loss': 0.0,
                           'soluble_part': 0.0,
                           'soluble_zapas': 0.0,
                           'liquid_flow': 0.0}, 
             'bottom_layer': {'activity_concentration': 0, 
                              'zapas': 0},
             'decay_constant': pu241_decay_constant,
             'infiltration_constant': am241_infiltration_constant,
             'inflow': 0.0,
             'name': 'Pu-241'}
    self.isotopes = [self.cs137, self.am241, self.pu241]

  def calculateAngle(self):
    ''' Calculate maximum angle of inclination of the ground surface 
        to the cell from a surrounding cells'''
    max_diff = 0
    for cell in self.up_way_cells:
      if abs(cell.x-self.x) <= 1 and abs(cell.y-self.y) <= 1 and cell.land_type != 'w':
        diff = abs(self.elevation-cell.elevation)
        if diff > max_diff:
          max_diff = diff
          if abs(cell.x-self.x) + abs(cell.y-self.y) == 1 :
            length = 100
          else : 
            length = 141.4
    if self.output_cell :
      if abs(self.elevation - self.output_cell.elevation) > max_diff and self.output_cell.land_type != 'w':
        max_diff = abs(self.elevation - self.output_cell.elevation)
        if abs(self.output_cell.x-self.x) + abs(self.output_cell.y-self.y) == 1 :
          length = 100
        else : 
          length = 141.4
    if max_diff > 0 :
      self.uklon = max_diff / length
      self.angle = max_diff/(math.sqrt(length**2+max_diff**2))

  def calculateLS(self) :
    self.calculateAngle()
    # TODO Попробовать с этим вычислением и по-умолчанию
    #self.model_LS = ((1+len(self.up_way_cells))*self.size/22.1)**0.4 * (self.angle*0.01745/0.09)**1.4 

  def calculate_max_outflow_capicity(self):
    if self.uklon > 0:
      tau_0 = 1005 * 9.8 * (0.003*np.random.randn() + 0.033) * self.uklon 
      g_s = 3.912 * tau_0 * (tau_0 - (0.03*np.random.randn() + 0.22)) / 9.8**2
      s_w = 4 * (2*np.random.randn() + 30) * (0.003*np.random.randn() + 0.033)
      self.max_outflow_capicity = g_s * s_w * (2*np.random.randn()+20)
    else:
      self.max_outflow_capicity = 0

  def findInputCells(self, catchment):
    stack = [self]
    added = []
    self.input_cells = []
    while stack :
      current = stack.pop()
      added.append(current)
      for cell in catchment :
        if abs(cell.x-self.x) <= 1 and abs(cell.y-self.y) <= 1 and cell.elevation > self.elevation :
          if (cell not in added) and (cell not in stack) and cell != current and cell != self :
            stack.append(cell)
            self.input_cells.append(cell)
    
  def findOutputCell(self, catchment):
    max_diff = 0
    self.output_cell = self.output_cell_storage = None
    for cell in catchment :
      if abs(cell.x-self.x) <= 1 and abs(cell.y-self.y) <= 1 and cell.elevation < self.elevation :
        if cell != self :
          self.output_cell = cell
          self.output_cell_storage = cell
          max_diff = self.elevation - cell.elevation

  def findUpWayCells(self, catchment) :
    stack = [self]
    added = []
    while stack :
      current = stack.pop()
      added.append(current)
      for cell in catchment :
        if cell.output_cell == current :
          if (cell not in added) and (cell not in stack) and cell != current and cell != self :
            stack.append(cell)
            self.up_way_cells.append(cell)
    self.up_way_cells_storage = self.up_way_cells[:]

  def calculateSoilLoss(self) :
    if self.land_type != 'w' :
      self.soil_loss = self.model_R * self.model_K * self.model_LS * self.model_C * self.model_P

  def calculateActivityLoss(self) :
    if self.land_type != 'w' :
      for isotope in self.isotopes :
        isotope['top_layer']['activity_loss'] = self.soil_loss * isotope['top_layer']['activity_concentration']

  def __repr__(self):
    return ("\n x=%d; y=%d; h=%d" % (self.x, self.y, self.elevation))

  def calculate_liquid_outflow(self, w_e=20.0, n_fl=3):
    fool_capicity = 25 + 2*np.random.randn() # 25 l -- volume of soil per 1 sq.m. wetted in the event
    w_fl = self.w0 + w_e - fool_capicity
    if w_fl < 0 :
      w_fl = 0
    for isotope in self.isotopes:
      if isotope['top_layer']['soluble_part'] > 0 :
        isotope['top_layer']['liquid_flow'] = n_fl * w_fl * ((isotope['top_layer']['zapas'] * isotope['top_layer']['soluble_part'])/(self.w0 + w_e))

  def infiltration(self):
    if self.land_type != 'w':
      for isotop in self.isotopes:
        isotop['top_layer']['zapas'] -= isotop['infiltration_constant'] * isotop['top_layer']['zapas']
        isotop['bottom_layer']['zapas'] += isotop['infiltration_constant'] * isotop['top_layer']['zapas']
        # and liquid flow out
        isotop['top_layer']['zapas'] -= isotop['top_layer']['liquid_flow']

  def radioactive_decay(self):
    for isotop in self.isotopes:
      isotop['top_layer']['zapas'] -= isotop['decay_constant'] * isotop['top_layer']['zapas']
      isotop['bottom_layer']['zapas'] -= isotop['decay_constant'] * isotop['bottom_layer']['zapas']
      if isotop['name'] == 'Pu-241':
        doter_pu241_top = isotop['decay_constant'] * isotop['top_layer']['zapas']
        doter_pu241_bottom = isotop['decay_constant'] * isotop['bottom_layer']['zapas']
    self.isotopes[1]['top_layer']['zapas'] += doter_pu241_top * 1.27e11 # specific activity of Am-241
    self.isotopes[1]['bottom_layer']['zapas'] += doter_pu241_bottom * 1.27e11

  def deposition(self):
    for isotop in self.isotopes:
      isotop['top_layer']['zapas'] += isotop['inflow']
      isotop ['inflow'] = 0

  def recalculation_activity_concentration(self):
    if self.land_type != 'w':
      for isotop in self.isotopes:
        isotop['top_layer']['activity_concentration'] = isotop['top_layer']['zapas'] / (self.size**2 *self.soil_density*(0.2*10*10))
        isotop['bottom_layer']['activity_concentration'] = isotop['bottom_layer']['zapas'] / (self.size**2 *self.soil_density*(1.8*10*10))
        isotop['top_layer']['soluble_zapas'] = isotop['top_layer']['soluble_part'] * isotop['top_layer']['zapas']

  def pond_clearing(self):
    ''' Удаляем весь смытый цезий со дна водоема, чтобы считать вынос только за один год
    '''
    for isotop in self.isotopes:
      isotop['top_layer']['activity_concentration'] = 0.0
      isotop['top_layer']['zapas'] = 0.0

  def year_redistribution(self, cumulative=True):
    self.infiltration()
    self.deposition()
    self.radioactive_decay()
    self.recalculation_activity_concentration()

class CatchmentArea():
  '''
     Класс, описывающий водосбор в целом, содержит catchment -- массив с отдельными ячейками модели
     Содержит функции подготовки ячеек к расчету стока и собственно функции рассчета стока
  '''
  def __init__(self, datafile=datafile):
    self.catchment = []
    with open(datafile) as f:
      reader = csv.reader(f)
      for row in reader:
        SC = SoilCell(int(row[0]), int(row[1]), float(row[2]), str(row[3]), float(row[4]), float(row[5]), float(row[8]), float(row[9]), cs137_s_part=float(row[6]), w0=float(row[7]))
        self.catchment.append(SC)
  
      for cell in self.catchment :
        cell.findOutputCell(self.catchment)
      for cell in self.catchment :
        cell.findUpWayCells(self.catchment)
      for cell in self.catchment:
        cell.calculateAngle()
        cell.calculateLS()
        cell.calculateSoilLoss()
        cell.calculate_max_outflow_capicity()

  def sediment_flow(self):
    flow_run = True
    while flow_run:
      flow_run = False
      for sc in self.catchment:
        if len(sc.up_way_cells) == 0 and sc.output_cell != None and sc.land_type != 'w':
          if sc.sediment_inflow + sc.soil_loss < sc.max_outflow_capicity:
            outflo = sc.sediment_inflow + sc.soil_loss
            for isotope, output_isotope in zip(sc.isotopes, sc.output_cell.isotopes):
              temp_isotop = isotope['inflow'] + sc.soil_loss * isotope['top_layer']['activity_concentration']
              output_isotope['inflow'] += temp_isotop
              isotope['top_layer']['zapas'] -= temp_isotop
            sc.sediment_inflow = 0
          else:
            outflo = sc.max_outflow_capicity
            for isotope, output_isotope in zip(sc.isotopes, sc.output_cell.isotopes):
              if sc.soil_loss > sc.max_outflow_capicity or sc.sediment_inflow < 0.001:
                temp_isotop = sc.max_outflow_capicity * isotope['top_layer']['activity_concentration']
              else:
                temp_isotop = sc.soil_loss * isotope['top_layer']['activity_concentration'] + (sc.max_outflow_capicity - sc.soil_loss) * isotope['inflow']/sc.sediment_inflow
              output_isotope['inflow'] += temp_isotop
              isotope['top_layer']['zapas'] -= temp_isotop
            sc.sediment_inflow -= sc.max_outflow_capicity
          sc.output_cell.sediment_inflow += outflo
          sc.output_cell.up_way_cells.remove(sc)
          sc.output_cell = None
          flow_run = True

  def one_season_flow(self, cumulative=True):
    for cell in self.catchment :
      cell.sediment_inflow = 0 # Здесь обнуляю перенос массы почвы до ее подсчета
      if cell.land_type == 'w' and not cumulative:
        cell.pond_clearing()
    w_e = 1.2*np.random.randn() + 20
    n_fl = 0.3*np.random.randn() + 2
    n_fl = n_fl if n_fl > 0.5 else 0.5
    for cell in self.catchment :
      cell.calculate_liquid_outflow(w_e, n_fl)
    self.sediment_flow()    
    for cell in self.catchment :
      cell.year_redistribution()
      cell.up_way_cells = cell.up_way_cells_storage[:]
      if cell.output_cell_storage :
        cell.output_cell = cell.output_cell_storage

  def CalculateTotalArea(self):
    total_area = 0
    for cell in self.catchment:
      total_area += cell.size**2
    return total_area

  def CalculateLandArea(self):
    land_area = 0
    for cell in self.catchment:
      if cell.land_type != 'w':
        land_area += cell.size**2
    return land_area

  def CalculateTotalStockAm(self):
    total_stock = 0
    for cell in self.catchment:
      if cell.land_type != 'w':
        total_stock += cell.isotopes[1]['top_layer']['zapas'] + cell.isotopes[1]['bottom_layer']['zapas']
    return total_stock

  def CalculateTotalStock(self, i, layer='both'): # i - номер радионуклида в списке
    total_stock = 0
    for cell in self.catchment:
      if cell.land_type != 'w' and layer == 'both':
        total_stock += cell.isotopes[i]['top_layer']['zapas'] + cell.isotopes[i]['bottom_layer']['zapas']
      elif cell.land_type != 'w' and layer == 'top':
        total_stock += cell.isotopes[i]['top_layer']['zapas']
      if cell.land_type != 'w' and layer == 'bottom':
        total_stock += cell.isotopes[i]['bottom_layer']['zapas']
    return total_stock/1000

  def init_pu_contamination(self, pu241):
    for cell in self.catchment:
      cell.isotopes[1]['top_layer']['zapas'] = 0.0
      cell.isotopes[1]['top_layer']['activity_concentration'] = 0.0
      cell.isotopes[1]['bottom_layer']['zapas'] = 0.0
      cell.isotopes[1]['bottom_layer']['activity_concentration'] = 0.0
      if cell.land_type == 'w':
        cell.pu241 = {'top_layer': {'activity_concentration': 0.0,
                           'zapas': 0.0,
                           'soluble_part': 0.0, 
                           'soluble_zapas': 0.0, 
                           'activity_loss': 0.0,
                           'liquid_flow': 0.0}, 
             'bottom_layer': {'activity_concentration': (pu241 + 0.001*pu241*np.random.randn()) / (20 * cell.soil_density), # 20 - объем почвы верхнего 2-см слоя почвы 1х1 м
                              'zapas': 0}, 
             'decay_constant': pu241_decay_constant,
             'infiltration_constant': cell.isotopes[1]['infiltration_constant'],
             'inflow': 0.0,
             'name': 'Pu-241'
             }
        cell.pu241['bottom_layer']['zapas'] = cell.size**2 * cell.pu241['bottom_layer']['activity_concentration'] * cell.soil_density*(0.2*10*10)
      else :
        cell.pu241 = {'top_layer': {'activity_concentration': (pu241 + 0.001*pu241*np.random.randn()) / (20 * cell.soil_density), # 20 - объем почвы верхнего 2-см слоя почвы 1х1 м
                           'zapas': 0.0,
                           'soluble_part': 0.0, 
                           'soluble_zapas': 0.0, 
                           'activity_loss': 0.0,
                           'liquid_flow': 0.0}, 
             'bottom_layer': {'activity_concentration': 0, 
                              'zapas': 0}, 
             'decay_constant': pu241_decay_constant,
             'infiltration_constant': cell.isotopes[1]['infiltration_constant'],
             'inflow': 0.0,
             'name': 'Pu-241'
             }
        cell.pu241['top_layer']['zapas'] = cell.size**2 * cell.pu241['top_layer']['activity_concentration'] * cell.soil_density*(0.2*10*10)
      cell.isotopes[2] = cell.pu241

  def init_cs_contamination(self, total_area, land_area):
    total_cs = 0.0
    for cell in self.catchment:
      if cell.land_type != 'w':
        total_cs += cell.isotopes[0]['top_layer']['zapas'] + cell.isotopes[0]['bottom_layer']['zapas']
    cs_contamination = (total_area * total_cs/land_area)/total_area
    init_cs = cs_contamination/(1 - 2**((1986-2018)/30.1671)) # Запас Cs на момент выпадений Бк/м^2
    for cell in self.catchment:
      if cell.land_type == 'w':
        cell.cs137['bottom_layer']['activity_concentration'] = (init_cs + 0.001*init_cs*np.random.randn()) / (20 * cell.soil_density) # 20 - объем почвы верхнего 2-см слоя почвы 1х1 м
        cell.cs137['bottom_layer']['zapas'] = cell.size**2 * cell.cs137['bottom_layer']['activity_concentration'] * cell.soil_density*20
        cell.cs137['top_layer']['soluble_zapas'] = 0.0
        cell.cs137['top_layer']['activity_concentration'] = 0.0
        cell.cs137['top_layer']['soluble_zapas'] = 0.0
      else:
        cell.cs137['top_layer']['activity_concentration'] = (init_cs + 0.001*init_cs*np.random.randn()) / (20 * cell.soil_density) # 20 - объем почвы верхнего 2-см слоя почвы 1х1 м
        cell.cs137['top_layer']['zapas'] = cell.size**2 * cell.cs137['top_layer']['activity_concentration'] * cell.soil_density*20
        cell.cs137['top_layer']['soluble_zapas'] = cell.cs137['top_layer']['soluble_part'] * cell.cs137['top_layer']['zapas']
      cell.isotopes[0] = cell.cs137

  def calculate_pond_accumulation(self, total=False):
    sum_cs = 0
    sum_am = 0
    sum_lq = 0
    sed = 0
    for cell in self.catchment:
      if cell.land_type == 'w':
        sed += cell.sediment_inflow
        sum_cs += cell.isotopes[0]['top_layer']['zapas']
        sum_am += cell.isotopes[1]['top_layer']['zapas']
        if total:
          sum_cs += cell.isotopes[0]['bottom_layer']['zapas']
          sum_am += cell.isotopes[1]['bottom_layer']['zapas']
      else:
        sum_lq += cell.isotopes[0]['top_layer']['liquid_flow']
    return sed, sum_cs, sum_lq, sum_am

  def pond_deposition(self):
    num_w_cell = 0
    sum_lq = 0
    for cell in self.catchment:
      if cell.land_type == 'w':
        num_w_cell += 1
      else:
        sum_lq += cell.isotopes[0]['top_layer']['liquid_flow']
    add_from_liquid = sum_lq / num_w_cell
    for cell in self.catchment:
      if cell.land_type == 'w':
        cell.isotopes[0]['top_layer']['zapas'] += add_from_liquid

  def one_iteration(self, outfil=outfile):
    self.one_season_flow()
    mod_data = self.calculate_pond_accumulation()
    out_string = '%d,%.3e,%.2e,%.2e,%.2e\n' % (i, mod_data[0], mod_data[1], mod_data[2], mod_data[3])
    with open(outfile, 'a') as outfl:
      outfl.write(out_string)

  def many_iterations(self, period, outfil=outfile, cumulative=True):
    disolved_cs137 = 0
    disolved_am241 = 0
    sediments = 0
    # Calculate initial stock of ^{241}Pu based on the current (2018) stock of ^{241}Am #
    total_area = self.CalculateTotalArea()
    land_area = self.CalculateLandArea()
    total_stock_am = self.CalculateTotalStockAm()
    total_stock_am = total_area * total_stock_am/land_area
    m_am = total_stock_am / 1.27e11 # specific activity of ^{241}Am
    m_pu241_init = m_am/(1 - 2**((1986-2018)/14.467))
    m_pu241_dens = m_pu241_init/total_area # g m^{-2}
    self.init_pu_contamination(m_pu241_dens)
    self.init_cs_contamination(total_area, land_area)
    for i in range(period):
      self.one_season_flow(cumulative=cumulative)
      mod_data = self.calculate_pond_accumulation()
      if cumulative:
        disolved_cs137 += mod_data[2]
        disolved_cs137 -= cs137_decay_constant * disolved_cs137
      else:
        disolved_cs137 = mod_data[2]
      ##disolved_am241 += mod_data[3]
      sediments += mod_data[0]
    if cumulative:
      # полный запас радионуклидов с учетом первоначальных выпадений total=True
      mod_data = self.calculate_pond_accumulation(total=True)
    else:
      mod_data = self.calculate_pond_accumulation()
    
    if outfil:
      out_string = '%d,%.3e,%.2e,%.2e,%.2e\n' % (i, sediments, mod_data[1], disolved_cs137, mod_data[3])
      with open(outfile, 'a') as outfl:
        outfl.write(out_string)
    else:
      return sediments, mod_data[1], disolved_cs137, mod_data[3]

  def surface_redistribution(self, period):
    contamination = np.array([])
    # Calculate initial stock of ^{241}Pu based on the current (2018) stock of ^{241}Am #
    total_area = self.CalculateTotalArea()
    land_area = self.CalculateLandArea()
    total_stock_am = self.CalculateTotalStockAm()
    total_stock_am = total_area * total_stock_am/land_area
    m_am = total_stock_am / 1.27e11 # specific activity of ^{241}Am
    m_pu241_init = m_am/(1 - 2**((1986-2018)/14.467))
    m_pu241_dens = m_pu241_init/total_area # g m^{-2}
    self.init_pu_contamination(m_pu241_dens)
    self.init_cs_contamination(total_area, land_area)
    for i in range(period):
      self.one_season_flow(cumulative=True)
      self.pond_deposition()
    for cell in self.catchment:
      k = 1 / (cell.size**2 * 1000) # factor for recalculating in kBq m^{-2}
      result = np.array([period, cell.x, cell.y, cell.land_type,
                         cell.isotopes[0]['top_layer']['zapas'] * k,
                         cell.isotopes[0]['bottom_layer']['zapas'] * k,
                         cell.isotopes[1]['top_layer']['zapas'] * k,
                         cell.isotopes[1]['bottom_layer']['zapas'] * k]).reshape(1, -1)
      try:
        contamination = np.append(contamination, result, axis=0)
      except:
        contamination = result.copy()
    return contamination

if __name__ == '__main__':
  #try :
  print(sys.argv[1])
  n_iterations = int(sys.argv[2])
  if sys.argv[1] == '1':
    with open(outfile, 'w') as outfl:
      outfl.write('iteration,solid_sediment_kg,solid_cs137_Bq,liquid_cs137_bq,solid_am241_bq\n')
    for i in tqdm_notebook(range(n_iterations)):
      catchment = CatchmentArea()
      catchment.one_iteration()
  else: 
      #try: 
    period = int(sys.argv[1])
    with open(outfile, 'w') as outfl:
      outfl.write('iteration,solid_sediment_kg,solid_cs137_Bq,liquid_cs137_bq,solid_am241_bq\n')
    for i in range(n_iterations):
      catchment = CatchmentArea()
# Расчет твердого стока только за один год: cumulative=False
      catchment.many_iterations(period, cumulative=True)
      #except:
      #  print("Can't interpret period {}".format(sys.argv[1]))
  #except :
  #  print('RTFM')
