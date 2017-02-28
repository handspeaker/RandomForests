objects = MnistPreProcess.o Node.o RandomForest.o Sample.o Tree.o main.o

INCLUDE_DIRS := /usr/local/include
LIBRARY_DIRS := /usr/local/lib
#LIBRARY_DIRS :=

LDFLAGS := $(foreach librarydir, $(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
$(foreach library,$(LIBRARIES),-l$(library))
COMMON_FLAGS := $(foreach includedir, $(INCLUDE_DIRS), -I$(includedir))
CXXFLAGS := -g $(COMMON_FLAGS)

RandomForests : $(objects)
	g++ -o RandomForests $(objects) $(CXXFLAGS) $(LDFLAGS)

.PHONY : clean
clean :
	-rm RandomForests $(objects)
