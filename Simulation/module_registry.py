# module_registry.py
import importlib
import inspect
import logging
import os
import sys
from abc import ABC
from typing import Dict, Type, Any, List, Optional
from pydantic import BaseModel, ValidationError
from utils import FieldManager

logger = logging.getLogger(__name__)

class ModuleDependencyError(Exception):
    """Raised when a module's dependency is not met."""
    pass

class ModuleInitializationError(Exception):
    """Raised when a module fails to initialize."""
    pass

class ModuleConfigurationError(Exception):
    """Raised when a module's configuration is invalid."""
    pass

class ModuleRegistry:
    """
    A robust module registry for managing physics modules in the DPF simulation.
    """
    def __init__(self):
        self.modules: Dict[Type, Dict[str, Any]] = {}  # Class -> {config_schema, metadata, dependencies}

    def register(self, module_class: Type, config_schema: Optional[Type[BaseModel]] = None, dependencies: Optional[List[Type]] = None, field_manager_required: bool = False, **metadata):
        """
        Registers a module class with the registry.

        Args:
            module_class: The class of the module to register.
            config_schema: An optional Pydantic BaseModel for validating the module's configuration.
            dependencies: An optional list of module classes that this module depends on.
            field_manager_required: A boolean indicating if the module requires a FieldManager.
            **metadata: Additional metadata about the module (e.g., author, description, version).
        """
        if not inspect.isclass(module_class):
            raise TypeError("module_class must be a class.")

        # Check if module_class is a subclass of ABC
        if not issubclass(module_class, ABC):
            raise TypeError("module_class must be a subclass of ABC (Abstract Base Class).")

        self.modules[module_class] = {
            'config_schema': config_schema,
            'metadata': metadata,
            'dependencies': dependencies or [],
            'field_manager_required': field_manager_required
        }
        logger.info(f"Registered module: {module_class.__name__}")

    def discover_plugins(self, package_name: str):
        """
        Dynamically discovers and registers modules within a specified package.

        This method imports all modules within the given package and attempts to
        register any classes that inherit from PhysicsModule.

        Args:
            package_name: The name of the Python package to scan for plugins.
        """
        try:
            package = importlib.import_module(package_name)
            package_path = os.path.dirname(package.__file__)

            for module_name in os.listdir(package_path):
                if module_name.startswith('_') or not module_name.endswith('.py'):
                    continue  # Skip private modules and non-Python files

                module_path = os.path.join(package_path, module_name)
                if os.path.isfile(module_path):
                    module_name = module_name[:-3]  # Remove '.py' extension
                    try:
                        module = importlib.import_module(f'{package_name}.{module_name}')
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, PhysicsModule) and obj != PhysicsModule:
                                self.register(obj)
                    except ImportError as e:
                        logger.warning(f"Could not import module {module_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error during plugin discovery in {module_name}: {e}")
        except ImportError as e:
            logger.error(f"Could not import package {package_name}: {e}")
        except Exception as e:
            logger.error(f"Error during plugin discovery: {e}")

    def create(self, module_class: Type, config: Optional[Dict] = None, created_modules: Optional[Dict[Type, Any]] = None, field_manager: Optional[FieldManager] = None) -> Any:
        """
        Creates an instance of a module, handling dependencies and configuration.

        Args:
            module_class: The class of the module to create.
            config: An optional configuration dictionary for the module.
            created_modules: A dictionary of already created modules to resolve dependencies.
            field_manager: An optional FieldManager object to pass to the module.

        Returns:
            An instance of the module class.

        Raises:
            ValueError: If the module is unknown.
            ModuleDependencyError: If a dependency is not met.
            ModuleConfigurationError: If the configuration is invalid.
            ModuleInitializationError: If the module fails to initialize.
        """
        if module_class not in self.modules:
            raise ValueError(f"Unknown module: {module_class.__name__}. Please ensure the module is registered.")

        # Handle dependencies
        created_modules = created_modules or {}
        dependencies = self.modules[module_class]['dependencies']
        for dep_class in dependencies:
            if dep_class not in created_modules:
                logger.info(f"Creating dependency {dep_class.__name__} for {module_class.__name__}")
                created_modules[dep_class] = self.create(dep_class, config=None, created_modules=created_modules, field_manager=field_manager)

        # Validate configuration
        config_schema = self.modules[module_class]['config_schema']
        if config and config_schema:
            try:
                validated_config = config_schema(**config)
                config = validated_config.model_dump()
            except ValidationError as e:
                # Provide user-friendly error messages
                error_messages = "\n".join([f"  {err['loc']}: {err['msg']}" for err in e.errors()])
                raise ModuleConfigurationError(f"Invalid configuration for module {module_class.__name__}:\n{error_messages}")

        # Instantiate the module
        try:
            # Check if the module requires a FieldManager
            if self.modules[module_class]['field_manager_required']:
                if field_manager is None:
                    raise ModuleDependencyError(f"Module {module_class.__name__} requires a FieldManager.")
                if config:
                    if "field_manager" in inspect.signature(module_class).parameters:
                        module = module_class(**config, field_manager=field_manager)
                    else:
                        module = module_class(**config)
                else:
                    if "field_manager" in inspect.signature(module_class).parameters:
                        module = module_class(field_manager=field_manager)
                    else:
                        module = module_class()
            else:
                if config:
                    module = module_class(**config)
                else:
                    module = module_class()
            logger.info(f"Created module: {module_class.__name__}")
            return module
        except TypeError as e:
            raise ModuleInitializationError(f"Error instantiating module {module_class.__name__}: {e}")

    def initialize_module(self, module: Any):
        """
        Initializes a module (e.g., load data, connect to resources).

        Args:
            module: The module instance to initialize.
        """
        if hasattr(module, 'initialize'):
            try:
                module.initialize()
                logger.info(f"Initialized module: {module.__class__.__name__}")
            except Exception as e:
                raise ModuleInitializationError(f"Error initializing module {module.__class__.__name__}: {e}")

    def finalize_module(self, module: Any):
        """
        Finalizes a module (e.g., close connections, save data).

        Args:
            module: The module instance to finalize.
        """
        if hasattr(module, 'finalize'):
            try:
                module.finalize()
                logger.info(f"Finalized module: {module.__class__.__name__}")
            except Exception as e:
                raise ModuleInitializationError(f"Error finalizing module {module.__class__.__name__}: {e}")

    def get_module_metadata(self, module_class: Type) -> Dict[str, Any]:
        """
        Returns the metadata for a module.

        Args:
            module_class: The class of the module.

        Returns:
            A dictionary containing the module's metadata.

        Raises:
            ValueError: If the module is unknown.
        """
        if module_class not in self.modules:
            raise ValueError(f"Unknown module: {module_class.__name__}")
        return self.modules[module_class]['metadata']
