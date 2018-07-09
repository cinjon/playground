import abc

class Schedule(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def step(self):
    """Step the wrapped_value"""
    raise NotImplementedError

  @property
  def wrapped_value(self):
    """Access wrapped value without side-effect"""
    raise NotImplementedError

  @property
  def value(self):
    """Step schedule and return the old value"""
    return_value = self.wrapped_value
    self.step()
    return return_value

  def __repr__(self):
    return str(self.wrapped_value)


class LinearSchedule(Schedule):
  def __init__(self, min_val=0.0, max_val=1.0,
               num_steps=1000, invert=False):
    super(LinearSchedule, self).__init__()

    self.min_val = min_val
    self.max_val = max_val
    self.num_steps = num_steps
    self.incr = float(max_val - min_val) / num_steps
    self.invert = invert
    if self.invert:
      self.incr *= -1.0

    self.val = self.max_val if invert else self.min_val

  @property
  def wrapped_value(self):
    return self.val

  def step(self):
    self.val = self.wrapped_value + self.incr
    if self.invert:
      self.val = max(self.val, self.min_val)
    else:
      self.val = min(self.val, self.max_val)
