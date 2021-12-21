/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */
import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;
import java.util.*;

public class BookDetails extends JFrame implements ActionListener{
    
    private JTable table;
    private JPanel controlPanel;
    private JTextField Search;
    private JButton b1,b2,b3;
    
    public static void main(String[] args) {
        new BookDetails().setVisible(true);
    }
    
    public BookDetails(){
        
        setBounds(350,200,890,475);
        controlPanel = new JPanel();
        controlPanel.setLayout(null);
        controlPanel.setBackground(Color.WHITE);
        setContentPane(controlPanel);
        
        JScrollPane scrollBar = new JScrollPane();
        scrollBar.setBounds(49,133,771,282);
        controlPanel.add(scrollBar);
        
        JTable table = new JTable();
        // WE NEED TO ADD THE TABLE TO THE SCROLLBAR 
//        table.addActionListener(ae);

        table.addMouseListener(new MouseAdapter(){
            public void mouseClicked(MouseEvent arg0){
                //@Override
// get Modal returns all the values of the current tablle
            int row = table.getSelectedRow();
            Search.setText(table.getModel().getValueAt(row, 1).toString());              } 
          });
            table.setBackground(new Color(240,240,255));
            table.setForeground(new Color(0,0,0));
            scrollBar.setViewportView(table);
            
             // add an image to a button
            JButton b1 = new JButton("Search");
            b1.addActionListener(this);
            ImageIcon i1 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/eight.png"));
            Image i2 = i1.getImage().getScaledInstance(200, 200, Image.SCALE_DEFAULT );
            ImageIcon i3 = new ImageIcon();
            b1.setIcon(i3);
            b1.setBounds(564,89,138,33);
            controlPanel.add(b1);
            
            JButton b2 = new JButton("Delete");
            b2.addActionListener(this);
            ImageIcon i4 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/eight.png"));
            Image i5 = i4.getImage().getScaledInstance(200, 200, Image.SCALE_DEFAULT );
            ImageIcon i6 = new ImageIcon();
            b2.setIcon(i6);
            b2.setBounds(712,89,138,33);
            controlPanel.add(b2);
            
            JTextField Search = new JTextField();
            Search.setBounds(189,89,357,33);
            controlPanel.add(Search);
            Search.setColumns(10);
            
            JLabel l1 = new JLabel("Back");
            l1.addMouseListener(new MouseAdapter(){
                //@Override         
                public void mouseClicked(MouseEvent e){
                    setVisible(false);
                    new Home().setVisible(true);
                }
            });
            
            ImageIcon i7 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/tenth.png"));
            Image i8 = i7.getImage().getScaledInstance(20, 20, Image.SCALE_DEFAULT );
            ImageIcon i9 = new ImageIcon(i8);  
            l1.setIcon(i9);
            l1.setBounds(97,89,72,33);
            controlPanel.add(l1);
            
            JPanel panel = new JPanel();
            panel.setBorder(new TitledBorder(new LineBorder(new Color(128,128,0),2),"Book Details",
                TitledBorder.LEADING,TitledBorder.TOP,null, new Color(34,129,34)));
        // leading top is top left
            panel.setBounds(31, 46, 840, 386);
            panel.setBackground(Color.WHITE);
            controlPanel.add(panel);
             
    }
        public void actionPerformed(ActionEvent ae)
        {
            try
            {
                conn con = new conn();
                if (ae.getSource() == b2){
                    String sql = "delete from book where name ='"+Search.getText()+"'";
                    PreparedStatement st = con.c.prepareStatement(sql);
                    
                    JDialog.setDefaultLookAndFeelDecorated(true);
                    int response = JOptionPane.showConfirmDialog(null,"Do you want to continue","Confirm",
                            JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE);
                    if (response == JOptionPane.NO_OPTION){}
                    else if (response == JOptionPane.YES_OPTION)
                    {
                        int rs = st.executeUpdate();
                        JOptionPane.showMessageDialog(null, "Delete !!");
                    }                
                    else if(response == JOptionPane.CLOSED_OPTION){
                    }st.close();
                    
                }
                if (ae.getSource() == b1)
                {
                
                }
                
                
            }
            
            catch(Exception e)
            {
            
            }
        
        }
    
}
