### Examples

In the following examples, I will achive similar results using a lot of different approachs to
show the flexibility of the library.

The idea is that you can use the approach that you feel more confortable with and develop your own
style of coding. For example in the first example I put training logic inside the aggregate, but
some people may say that the domain should not know about the training logic, and should care only
about it's data, so in the following examples the training logic is docupled is several ways
from the aggregate, like using commands or creating a service.