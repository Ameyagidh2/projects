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
import java.sql.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.event.*;
import net.proteanit.sql.DbUtils;

public class Statistics extends JFrame {
    private JPanel panel;
    private JTable table1,table2;
   
    public static void main(String[] args) {
        new Statistics().setVisible(true);
    }
    
    public void issue_book(){
    try{
        conn con = new conn();
        String sql = "select * from issueBook";
        PreparedStatement st = con.c.prepareStatement(sql);
        ResultSet rs = st.executeQuery();
        table1.setModel(DbUtils.resultSetToTableModel(rs));
    
    }catch(Exception e){
    }
        
    }
    public void return_book(){
    try{
        conn con = new conn();
        String sql = "select * from returnBook";
        PreparedStatement st = con.c.prepareStatement(sql);
        ResultSet rs = st.executeQuery();
        table1.setModel(DbUtils.resultSetToTableModel(rs));
    
    }catch(Exception e){
    }
        
    }
   
    
    public Statistics(){
        setBounds(400,200,860,600);
        panel  = new JPanel();
        panel.setBackground(Color.WHITE);
        setContentPane(panel);
        panel.setLayout(null);
        
        JScrollPane scrollbar1 = new JScrollPane();
        scrollbar1.setBounds(40,51,708,217);
        panel.add(scrollbar1);
        
        JScrollPane scrollbar2 = new JScrollPane();
        scrollbar2.setBounds(40,321,708,217);
        panel.add(scrollbar2);
        
        table1 = new JTable();
        scrollbar1.setViewportView(table1);
        
        table2 = new JTable();
        scrollbar1.setViewportView(table2);
        
        JPanel panel_1 = new JPanel();
        panel_1.setBorder(new TitledBorder(new LineBorder(new Color(0, 204, 153), 2, true), "Issue-Book-Details",
		TitledBorder.RIGHT, TitledBorder.TOP, null, new Color(0, 102, 51)));
	panel_1.setBounds(22, 29, 741, 240);
        panel_1.setBackground(Color.WHITE);
	panel.add(panel_1);
        
        JPanel panel_2 = new JPanel();
        panel_2.setBorder(new TitledBorder(new LineBorder(new Color(0, 204, 153), 2, true), "Return-Book-Details",
		TitledBorder.RIGHT, TitledBorder.TOP, null, new Color(0, 102, 51)));
	panel_2.setBounds(22, 299, 741, 240);
        panel_2.setBackground(Color.WHITE);
	panel.add(panel_2);
        
        JLabel l1 = new JLabel("Back");
        l1.addMouseListener(new MouseAdapter(){
            public void mouseClicked(MouseEvent arg0)
            {
                setVisible(false);
                new Home().setVisible(true);            
            }});   
       
        ImageIcon i1 = new ImageIcon(ClassLoader.getSystemResource("library/management/icons/tenth.png"));  
        Image i2 = i1.getImage().getScaledInstance(20, 20, Image.SCALE_DEFAULT);
        ImageIcon i3 = new ImageIcon(i2);
        l1.setIcon(i3);
        l1.setBounds(770,27,76,27);
        panel.add(l1);
        
        issue_book();
        return_book();
        
    }
    
}
