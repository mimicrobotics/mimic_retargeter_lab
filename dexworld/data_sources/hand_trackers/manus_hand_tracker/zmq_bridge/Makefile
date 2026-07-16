CXX      := g++
CXXFLAGS := -g --std=c++17
INCLUDE  := -I../ManusSDK/include -I.
LDFLAGS  := -L../ManusSDK/lib -lManusSDK_Integrated -lpthread -lzmq -Wl,-rpath=../ManusSDK/lib

OBJ_DIR  := ./objects
TARGET   := ManusZmqBridge.out

all: build $(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(TARGET): $(OBJ_DIR)/ManusZmqBridge.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

-include $(OBJ_DIR)/ManusZmqBridge.d

.PHONY: all build clean

build:
	@mkdir -p $(OBJ_DIR)

clean:
	-@rm -f $(TARGET)
	-@rm -rvf $(OBJ_DIR)/*
