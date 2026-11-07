#include <OpenSim/OpenSim.h>
#include <iostream>

int main() {
    try {
        // 1. Load the model
        OpenSim::Model model("your_model.osim");
        
        // 2. Instantiate the IK Tool
        OpenSim::InverseKinematicsTool ikTool;
        
        // 3. Connect the tool to the model and the marker data
        ikTool.setModel(model);
        ikTool.setMarkerDataFileName("motion_capture_data.trc");
        
        // 4. Set time bounds (read from your data or hardcoded)
        ikTool.setStartTime(0.0);
        ikTool.setEndTime(5.0);
        
        // 5. Specify the output file name where q values will be written
        ikTool.setOutputMotionFileName("ik_output_q_values.mot");
        
        // Optional: If you have an IK tasks file defining marker weights
        // ikTool.getIKTaskSet().createFromASTorXMLFile("ik_tasks.xml");
        
        // 6. Run the tool
        std::cout << "Running Inverse Kinematics..." << std::endl;
        ikTool.run();
        std::cout << "IK finished successfully! Results saved to ik_output_q_values.mot" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "OpenSim Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


