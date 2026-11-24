{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :show-inheritance:
{#- The base class of the models and the base class of each kind of part own the registry of their subclasses.
    Their pages also show the methods of the registry mixin, which are how a checkpoint rebuilds models and
    parts. The page of a model family also shows what the family inherits from the base class of the models,
    since a model is trained, saved and loaded through those methods. Everything inherited from nn.Module
    stays out. #}
{%- if ("registry" in members and "registry" not in inherited_members) or "part_kinds" in members %}
   :inherited-members: Module
{%- endif %}
